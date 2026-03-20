from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
import os
import dotenv
import sendgrid
from sendgrid.helpers.mail import Mail
import markdown
import time
from threading import Lock
from google_ai import ask_ai_model
from scholar_sync import (
    DEFAULT_SYNC_TTL_SECONDS,
    default_settings,
    parse_iso_datetime,
    sync_google_scholar_data,
)

dotenv.load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

# ============ DUAL-MODE SUPPORT: GitHub API or Local Files ============
USE_GITHUB_API = os.getenv('USE_GITHUB_API', 'false').lower() == 'true'

if USE_GITHUB_API:
    print("[ADMIN] Using GitHub API for file operations")
    from admin_github_ops import AdminGitHubOps
    file_ops = AdminGitHubOps()
else:
    print("[ADMIN] Using local file operations")
    from admin_file_ops import AdminFileOps
    file_ops = AdminFileOps(base_dir=os.getcwd())

# Import authentication module
from admin_auth import login_required, verify_password, log_admin_action
# ======================================================================

SCHOLAR_SYNC_LOCK = Lock()

def load_data():
    """Load fresh data from data.json"""
    try:
        all_data = file_ops.read_data()
        
        projects_data = [item for item in all_data if not item['id'].startswith('paper_')]
        research_data = [item for item in all_data if item['id'].startswith('paper_')]
        
        return all_data, projects_data, research_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], [], []


def load_settings():
    """Load settings from settings.json"""
    defaults = default_settings()

    try:
        return defaults | (file_ops.read_settings() or {})
    except Exception as e:
        print(f"Error loading settings: {e}")

    return defaults


@app.context_processor
def inject_global_template_settings():
    """Make shared branding settings available to every template."""
    settings = load_settings()
    return {
        'settings': settings,
        'profile_image_url': settings['profile_image_url'],
        'profile_image_fallback_url': settings['profile_image_fallback_url'],
        'scholar_publications': settings.get('scholar_publications', [])
    }


def load_item_by_id(item_id):
    """Load a single project/research item from data.json."""
    all_data, _, _ = load_data()
    return next((item for item in all_data if item.get('id') == item_id), None)


def maybe_sync_scholar_data(force=False):
    """Refresh Scholar metadata if the cache is stale or when forced."""
    settings = file_ops.read_settings() or {}
    settings = default_settings() | settings

    if not force and not settings.get('scholar_sync_enabled', True):
        return None

    ttl_seconds = int(settings.get('scholar_sync_ttl_seconds') or DEFAULT_SYNC_TTL_SECONDS)
    last_synced = parse_iso_datetime(settings.get('scholar_last_synced_at'))
    if not force and last_synced:
        age_seconds = time.time() - last_synced.timestamp()
        if age_seconds < ttl_seconds:
            return None

    if not SCHOLAR_SYNC_LOCK.acquire(blocking=False):
        return None

    try:
        refreshed_settings = default_settings() | (file_ops.read_settings() or {})
        if not force and refreshed_settings.get('scholar_last_synced_at'):
            last_synced = parse_iso_datetime(refreshed_settings.get('scholar_last_synced_at'))
            if last_synced and (time.time() - last_synced.timestamp()) < ttl_seconds:
                return None
        return sync_google_scholar_data(file_ops, refreshed_settings, force=force)
    finally:
        SCHOLAR_SYNC_LOCK.release()


@app.before_request
def refresh_scholar_metadata_if_needed():
    """Best-effort Scholar refresh for normal page loads."""
    if request.method not in {'GET', 'HEAD'}:
        return
    if request.path.startswith('/static') or request.path.startswith('/admin'):
        return
    maybe_sync_scholar_data(force=False)

@app.route('/')
def index():
    _, projects_data, research_data = load_data()
    projects_to_display = [
        next((p for p in projects_data if p['id'] == 'blockchain_ai'), None),
        next((p for p in projects_data if p['id'] == 'ml_audit_library'), None)
    ]
    projects_to_display = [p for p in projects_to_display if p]

    research_to_display = [
        next((r for r in research_data if r['id'] == 'paper_fair_explainable_credit'), None),
        next((r for r in research_data if r['id'] == 'paper_adversarial_cybersecurity'), None)
    ]
    research_to_display = [r for r in research_to_display if r]

    # Add details_page to the data
    for p in projects_to_display:
        p['details_page'] = 'project_' + p['id'] if not p['id'].startswith('project_') else p['id']

    for r in research_to_display:
        r['details_page'] = r['id']

    settings = load_settings()
    return render_template('index.html', projects=projects_to_display, research_papers=research_to_display, settings=settings)

@app.route('/ai_chat_modal/<item_id>')
def ai_chat_modal(item_id):
    # Find item from all_data
    item = load_item_by_id(item_id)
    if not item:
        return "Item not found", 404
    return render_template('ai_chat_modal.html', item_title=item['title'], item_id=item_id)

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    # Rate limiting: Track requests per session
    if 'ai_request_count' not in session:
        session['ai_request_count'] = 0
        session['ai_request_reset_time'] = time.time()
    
    # Reset counter every 5 minutes
    if time.time() - session.get('ai_request_reset_time', 0) > 300:
        session['ai_request_count'] = 0
        session['ai_request_reset_time'] = time.time()
    
    # Limit to 20 requests per 5 minutes
    if session['ai_request_count'] >= 20:
        return jsonify({'error': 'Rate limit exceeded. Please wait a few minutes before asking more questions.'}), 429
    
    session['ai_request_count'] += 1
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        item_id = data.get('item_id')
        query = data.get('query')

        # Input validation
        if not item_id or not query:
            return jsonify({'error': 'Missing item_id or query'}), 400
        
        # Sanitize inputs
        item_id = str(item_id).strip()
        query = str(query).strip()
        
        # Validate query length
        if len(query) > 1000:
            return jsonify({'error': 'Question is too long. Please keep it under 1000 characters.'}), 400
        
        if len(query) < 3:
            return jsonify({'error': 'Question is too short. Please provide more detail.'}), 400

        # Find the item from all_data
        item = load_item_by_id(item_id)

        if not item:
            return jsonify({'error': 'Item not found'}), 404

        # Build a comprehensive context
        context = f"Title: {item.get('title', '')}\n"
        context += f"Description: {item.get('description', '')}\n"
        if 'tech_stack' in item:
            context += f"Technology Stack: {', '.join(item.get('tech_stack', []))}\n"
        context += f"\nDetails: {item.get('details', '')}"
        
        # Call AI model with sanitized inputs
        ai_response = ask_ai_model(context, query)
        
        return jsonify({'response': ai_response})
    
    except Exception as e:
        # Log error for debugging
        print(f"Error in /ask_ai endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/projects')
def all_projects():
    # Load all projects
    _, projects_data, _ = load_data()
    # Filter out research papers just in case, though load_data already does this
    filtered_projects = [
        p for p in projects_data
        if not p['id'].startswith('paper_') and p.get('id') != 'ml_in_cybersec'
    ]
    
    for p in filtered_projects:
        p['details_page'] = 'project_' + p['id'] if not p['id'].startswith('project_') else p['id']
    return render_template('projects.html', projects=filtered_projects)

@app.route('/research')
def all_research():
    _, _, research_data = load_data()
    for r in research_data:
        r['details_page'] = r['id']
    return render_template('research.html', research_papers=research_data)

@app.route('/project/blockchain_ai')
def project_blockchain_ai():
    return render_template('blockchain_ai.html')

@app.route('/project/default')
def project_default():
    return render_template('project_default.html')



@app.route('/project/ml_audit')
def project_ml_audit():
    return render_template('project_ml_audit.html')

@app.route('/project/dataml')
def project_dataml():
    return render_template('project_dataml.html')

@app.route('/project/<project_id>')
def project_details(project_id):
    """Generic route for project details"""
    # Load data to check for custom content file
    project = load_item_by_id(project_id)
    
    if project and project.get('content_file'):
        filename = project['content_file']
        ext = os.path.splitext(filename)[1].lower()
        content_path = os.path.join(app.root_path, 'templates', 'content', filename)
        
        if os.path.exists(content_path):
            if ext == '.md':
                try:
                    with open(content_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()
                        # Use markdown to convert to html (safe default extensions)
                        try:
                            html_content = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])
                        except:
                            # Fallback if specific extensions fail
                            html_content = markdown.markdown(md_text)
                            
                        return render_template('project_dynamic_content.html', project=project, content=html_content)
                except Exception as e:
                    print(f"Error rendering markdown: {e}")
                    return render_template('project_dynamic_content.html', project=project, content=f"<div class='alert alert-danger'>Error displaying content: {str(e)}</div>")
            elif ext in ['.html', '.htm']:
                return render_template('project_dynamic_content.html', project=project, include_file='content/' + filename)

    # Check if a specific template exists for this project ID (Legacy support)
    # 1. Check for project_<id>.html
    template_name = f'project_{project_id}.html'
    template_path = os.path.join(app.root_path, 'templates', template_name)
    if os.path.exists(template_path):
        return render_template(template_name)
        
    # 2. Check for <id>.html (e.g. paper_adversarial_cybersecurity.html)
    template_name_direct = f'{project_id}.html'
    template_path_direct = os.path.join(app.root_path, 'templates', template_name_direct)
    if os.path.exists(template_path_direct):
         return render_template(template_name_direct)
    
    # Fallback to generic template with data from data.json
    
    # Fallback to generic template
    if not project:
        return "Project not found", 404
        
    return render_template('project_default.html', project=project)



@app.route('/research/default')
def research_default():
    return render_template('research_default.html')

@app.route('/paper/federated_credit')
def paper_federated_credit():
    return render_template('paper_federated_credit.html', paper=load_item_by_id('paper_federated_credit'))

@app.route('/paper/causal_health')
def paper_causal_health():
    return render_template('paper_causal_health.html', paper=load_item_by_id('paper_causal_health'))

@app.route('/paper/edge_ai_nas')
def paper_edge_ai_nas():
    return render_template('paper_edge_ai_nas.html', paper=load_item_by_id('paper_edge_ai_nas'))

@app.route('/paper/adversarial_cybersecurity')
def paper_adversarial_cybersecurity():
    return render_template('paper_adversarial_cybersecurity.html', paper=load_item_by_id('paper_adversarial_cybersecurity'))

@app.route('/paper/defi_risk')
def paper_defi_risk():
    return render_template('paper_defi_risk.html', paper=load_item_by_id('paper_defi_risk'))

@app.route('/paper/low_resource_med')
def paper_low_resource_med():
    return render_template('paper_low_resource_med.html', paper=load_item_by_id('paper_low_resource_med'))

@app.route('/paper/quantum_gnn')
def paper_quantum_gnn():
    return render_template('paper_quantum_gnn.html', paper=load_item_by_id('paper_quantum_gnn'))

@app.route('/paper/carbon_capture_opt')
def paper_carbon_capture_opt():
    return render_template('paper_carbon_capture_opt.html', paper=load_item_by_id('paper_carbon_capture_opt'))

@app.route('/paper/fair_explainable_credit')
def paper_fair_explainable_credit():
    return render_template('paper_fair_explainable_credit.html', paper=load_item_by_id('paper_fair_explainable_credit'))

@app.route('/api_docs')
def api_docs():
    return render_template('api_docs.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_body = request.form['message']

        message = Mail(
            from_email='techbidmarketplace@gmail.com',
            to_emails='techbidmarketplace@gmail.com',
            subject=f"New Message from {name} ({email})",
            html_content=f'<strong>Name:</strong> {name}<br><strong>Email:</strong> {email}<br><strong>Message:</strong><br>{message_body}'
        )
        message.reply_to = email

        try:
            sg = sendgrid.SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
            response = sg.send(message)
            flash('Your message has been sent successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            print(e)
            flash(f'Error sending message.', 'error')

    return render_template('index.html')

# ============ ADMIN ROUTES ============

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        password = request.form.get('password')
        
        if verify_password(password):
            session['admin_logged_in'] = True
            session['last_activity'] = time.time()
            log_admin_action("Admin logged in")
            flash('Welcome to the admin panel!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid password. Please try again.', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.clear()
    log_admin_action("Admin logged out")
    flash('You have been logged out.', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard"""
    data = file_ops.read_data()
    settings = default_settings() | (file_ops.read_settings() or {})
    backups = file_ops.list_backups()
    images = file_ops.list_images()
    activities = file_ops.read_activity_log(max_entries=20)  # Get last 20 activities
    
    # Separate projects and research from list
    projects = [item for item in data if not item.get('id', '').startswith('paper_')]
    research = [item for item in data if item.get('id', '').startswith('paper_')]
    
    stats = {
        'total_projects': len(projects),
        'total_research': len(research),
        'total_backups': len(backups),
        'total_images': len(images)
    }
    
    # Create data dict for template
    data_dict = {
        'projects': projects,
        'research': research
    }
    
    return render_template('admin_dashboard.html', 
                         data=data_dict, 
                         stats=stats,
                         settings=settings,
                         backups=backups[:5],  # Show last 5 backups
                         activities=activities,
                         images=images)


@app.route('/admin/scholar/sync', methods=['POST'])
@login_required
def admin_sync_scholar():
    """Force a Google Scholar sync from the admin dashboard."""
    outcome = maybe_sync_scholar_data(force=True)
    if outcome and outcome.updated:
        log_admin_action(
            f"Updated Google Scholar data ({outcome.fetched_publications} publications, "
            f"{outcome.matched_papers} matched local papers)"
        )
        flash(
            f'Google Scholar sync complete: {outcome.fetched_publications} publications fetched, '
            f'{outcome.matched_papers} local papers updated.',
            'success'
        )
    else:
        flash(
            f'Google Scholar sync could not complete: {(outcome.error if outcome else "sync skipped")}.',
            'error'
        )
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/project/add', methods=['GET', 'POST'])
@login_required
def admin_add_project():
    """Add new project or research"""
    item_type = request.args.get('type', 'project')
    
    if request.method == 'POST':
        item_type = request.form.get('type', 'project')
        
        # Handle image upload
        image_file = request.files.get('image')
        image_filename = None
        if image_file and image_file.filename:
            image_filename = file_ops.upload_image(image_file)
        
        # Build project data
        # Clean up tech stack list (remove empty strings and whitespace)
        tech_stack = [t.strip() for t in request.form.get('tech_stack', '').split(',') if t.strip()]
        
        project_data = {
            'title': request.form.get('title'),
            'description': request.form.get('description'),
            'details': request.form.get('details', ''), # Detailed description for AI
            'tech_stack': tech_stack,
            'image': image_filename if image_filename else ''
        }
        
        
        id_prefix = 'paper_' if item_type == 'research' else ''
        content_string = request.form.get('content_markdown')
        
        if file_ops.add_project(project_data, id_prefix=id_prefix, content_string=content_string):
            log_admin_action(f"Added {item_type}: {project_data['title']}")
            flash(f'{item_type.capitalize()} "{project_data["title"]}" added successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash(f'Error adding {item_type}.', 'error')
    
    return render_template('admin_project_editor.html', mode='add', item_type=item_type)

@app.route('/admin/project/edit/<project_id>', methods=['GET', 'POST'])
@login_required
def admin_edit_project(project_id):
    """Edit existing project"""
    if request.method == 'POST':
        # Handle image upload
        image_file = request.files.get('image')
        image_filename = request.form.get('existing_image', '')
        
        if image_file and image_file.filename:
            # Delete old image if exists
            if image_filename:
                file_ops.delete_image(image_filename)
            image_filename = file_ops.upload_image(image_file)
        
        # Build project data
        project_data = {
            'title': request.form.get('title'),
            'description': request.form.get('description'),
            'details': request.form.get('details', ''), # Detailed description for AI
            'tech_stack': [t.strip() for t in request.form.get('tech_stack', '').split(',') if t.strip()],
            'image': image_filename if image_filename else '',
            'content_file': request.form.get('existing_content_file', '')
        }
        
        # Handle markdown content update
        content_string = request.form.get('content_markdown')
        if content_string is not None:
            filename = file_ops.save_content_string(project_id, content_string)
            if filename:
                 project_data['content_file'] = filename

        if file_ops.update_project(project_id, project_data):
            log_admin_action(f"Updated project: {project_data['title']}")
            flash(f'Project "{project_data["title"]}" updated successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Error updating project.', 'error')
    
    # GET request - show edit form
    project = file_ops.get_project(project_id)
    if not project:
        flash('Project not found.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    item_type = 'research' if project['id'].startswith('paper_') else 'project'
    
    # Load content for editing
    content_initial_value = ''
    if project.get('content_file'):
        filename = project['content_file']
        if filename.endswith('.md'):
            filepath = os.path.join(app.root_path, 'templates', 'content', filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content_initial_value = f.read()
                except Exception as e:
                    print(f"Error reading content file: {e}")

    return render_template('admin_project_editor.html', mode='edit', project=project, item_type=item_type, content_initial_value=content_initial_value)

@app.route('/admin/project/delete/<project_id>', methods=['POST'])
@login_required
def admin_delete_project(project_id):
    """Delete project"""
    project = file_ops.get_project(project_id)
    if project and file_ops.delete_project(project_id):
        log_admin_action(f"Deleted project: {project.get('title', project_id)}")
        flash('Project deleted successfully!', 'success')
    else:
        flash('Error deleting project.', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/upload/asset', methods=['POST'])
@login_required
def admin_upload_asset():
    """Handle generic asset upload for content editor"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = file_ops.upload_content_asset(file)
    if filename:
        return jsonify({'url': url_for('static', filename=filename)})
    
    return jsonify({'error': 'Upload failed'}), 500

@app.route('/admin/backup/create', methods=['POST'])
@login_required
def admin_create_backup():
    """Create backup of data.json"""
    backup_file = file_ops.backup_data()
    if backup_file:
        log_admin_action("Created backup")
        flash('Backup created successfully!', 'success')
    else:
        flash('Error creating backup.', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/backup/restore/<filename>', methods=['POST'])
@login_required
def admin_restore_backup(filename):
    """Restore from backup"""
    # Security check: ensure filename contains only safe characters
    if not filename.startswith('data_backup_') or not filename.endswith('.json') or '/' in filename or '\\' in filename:
        flash('Invalid backup filename.', 'error')
        return redirect(url_for('admin_dashboard'))
        
    backup_path = os.path.join(file_ops.backup_dir, filename)
    if os.path.exists(backup_path):
        if file_ops.restore_data(backup_path):
            log_admin_action(f"Restored backup: {filename}")
            flash('Data restored successfully from backup!', 'success')
        else:
            flash('Error restoring backup.', 'error')
    else:
        flash('Backup file not found.', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/backup/delete/<filename>', methods=['POST'])
@login_required
def admin_delete_backup(filename):
    """Delete a backup file"""
    if file_ops.delete_backup(filename):
        log_admin_action(f"Deleted backup: {filename}")
        flash('Backup deleted successfully!', 'success')
    else:
        flash('Error deleting backup.', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/image/delete/<filename>', methods=['POST'])
@login_required
def admin_delete_image(filename):
    """Delete an image file"""
    if file_ops.delete_image(filename):
        log_admin_action(f"Deleted image: {filename}")
        flash('Image deleted successfully!', 'success')
    else:
        flash('Error deleting image.', 'error')
    
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/settings/update', methods=['POST'])
@login_required
def admin_update_settings():
    """Update hero section metrics"""
    existing_settings = default_settings() | (file_ops.read_settings() or {})
    settings = existing_settings | {
        "production_models": request.form.get('production_models'),
        "model_uptime": request.form.get('model_uptime'),
        "client_projects": request.form.get('client_projects'),
        "years_experience": request.form.get('years_experience')
    }
    
    if file_ops.write_settings(settings):
        log_admin_action("Updated hero metrics")
        flash('Hero metrics updated successfully!', 'success')
    else:
        flash('Error updating settings.', 'error')
        
    return redirect(url_for('admin_dashboard'))
