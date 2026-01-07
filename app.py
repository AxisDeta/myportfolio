from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
import os
import dotenv
import sendgrid
from sendgrid.helpers.mail import Mail
import json
import time
from google_ai import ask_ai_model

dotenv.load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

# Load all data from data.json
with open('data.json') as f:
    all_data = json.load(f)

projects_data = [item for item in all_data if not item['id'].startswith('paper_')]
research_data = [item for item in all_data if item['id'].startswith('paper_')]

@app.route('/')
def index():
    projects_to_display = [
        next((p for p in projects_data if p['id'] == 'blockchain_ai'), None),
        next((p for p in projects_data if p['id'] == 'ml_audit'), None)
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

    return render_template('index.html', projects=projects_to_display, research_papers=research_to_display)

@app.route('/ai_chat_modal/<item_id>')
def ai_chat_modal(item_id):
    # Find item from all_data
    item = next((item for item in all_data if item['id'] == item_id), None)
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
        item = next((item for item in all_data if item['id'] == item_id), None)

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
    # Filter for specific projects only
    allowed_projects = ['blockchain_ai', 'ml_audit', 'dataml']
    filtered_projects = [p for p in projects_data if p['id'] in allowed_projects]
    
    for p in filtered_projects:
        p['details_page'] = 'project_' + p['id'] if not p['id'].startswith('project_') else p['id']
    return render_template('projects.html', projects=filtered_projects)

@app.route('/research')
def all_research():
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



@app.route('/research/default')
def research_default():
    return render_template('research_default.html')

@app.route('/paper/federated_credit')
def paper_federated_credit():
    return render_template('paper_federated_credit.html')

@app.route('/paper/causal_health')
def paper_causal_health():
    return render_template('paper_causal_health.html')

@app.route('/paper/edge_ai_nas')
def paper_edge_ai_nas():
    return render_template('paper_edge_ai_nas.html')

@app.route('/paper/adversarial_cybersecurity')
def paper_adversarial_cybersecurity():
    return render_template('paper_adversarial_cybersecurity.html')

@app.route('/paper/defi_risk')
def paper_defi_risk():
    return render_template('paper_defi_risk.html')

@app.route('/paper/low_resource_med')
def paper_low_resource_med():
    return render_template('paper_low_resource_med.html')

@app.route('/paper/quantum_gnn')
def paper_quantum_gnn():
    return render_template('paper_quantum_gnn.html')

@app.route('/paper/carbon_capture_opt')
def paper_carbon_capture_opt():
    return render_template('paper_carbon_capture_opt.html')

@app.route('/paper/fair_explainable_credit')
def paper_fair_explainable_credit():
    return render_template('paper_fair_explainable_credit.html')

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

if __name__ == "__main__":
    app.run(debug=True)