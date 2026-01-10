"""
Admin File Operations Module
Handles local file operations for portfolio management
Will be extended with GitHub API integration later
"""

import json
import os
import shutil
import time
from datetime import datetime
from werkzeug.utils import secure_filename

class AdminFileOps:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_file = os.path.join(base_dir, 'data.json')
        self.settings_file = os.path.join(base_dir, 'settings.json')
        self.static_dir = os.path.join(base_dir, 'static')
        self.backup_dir = os.path.join(base_dir, 'backups')
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
    
    # ============ SETTINGS OPERATIONS ============
    def read_settings(self):
        """Read settings.json"""
        try:
            if not os.path.exists(self.settings_file):
                # Return defaults if missing
                return {
                    "years_experience": "5",
                    "client_projects": "5",
                    "production_models": "8",
                    "model_uptime": "99.9"
                }
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading settings.json: {e}")
            return {}

    def write_settings(self, settings):
        """Write to settings.json"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error writing settings: {e}")
            return False

    # ============ DATA.JSON OPERATIONS ============
    
    def read_data(self):
        """Read data.json"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading data.json: {e}")
            return []  # Return empty list instead of dict
    
    def write_data(self, data):
        """Write to data.json"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing data.json: {e}")
            return False
    
    def backup_data(self):
        """Create backup of data.json"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f'data_backup_{timestamp}.json')
        try:
            shutil.copy2(self.data_file, backup_file)
            return backup_file
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    def restore_data(self, backup_file):
        """Restore data.json from backup"""
        try:
            shutil.copy2(backup_file, self.data_file)
            return True
        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self):
        """List all available backups with metadata"""
        try:
            backup_files = [f for f in os.listdir(self.backup_dir) if f.startswith('data_backup_')]
            backup_files.sort(reverse=True)  # Most recent first
            
            backups_info = []
            for filename in backup_files:
                # Parse timestamp from filename: data_backup_YYYYMMDD_HHMMSS.json
                try:
                    timestamp_str = filename.replace('data_backup_', '').replace('.json', '')
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Read backup to get summary
                    backup_path = os.path.join(self.backup_dir, filename)
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    projects = [item for item in data if not item.get('id', '').startswith('paper_')]
                    research = [item for item in data if item.get('id', '').startswith('paper_')]
                    
                    summary = f"{len(projects)} projects, {len(research)} research items"
                    
                    backups_info.append({
                        'filename': filename,
                        'timestamp': formatted_time,
                        'summary': summary,
                        'size': os.path.getsize(backup_path)
                    })
                except Exception as e:
                    # If parsing fails, add basic info
                    backups_info.append({
                        'filename': filename,
                        'timestamp': 'Unknown',
                        'summary': 'Unable to read backup',
                        'size': 0
                    })
            
            return backups_info
        except Exception as e:
            print(f"Error listing backups: {e}")
            return []
    
    def delete_backup(self, filename):
        """Delete a specific backup file"""
        try:
            # Security check: ensure filename is safe
            if not filename.startswith('data_backup_') or not filename.endswith('.json'):
                return False
            if '/' in filename or '\\' in filename or '..' in filename:
                return False
                
            backup_path = os.path.join(self.backup_dir, filename)
            if os.path.exists(backup_path):
                os.remove(backup_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting backup: {e}")
            return False
    
    def read_activity_log(self, max_entries=30):
        """Read recent activity log entries"""
        try:
            log_file = os.path.join(self.base_dir, 'admin_log.txt')
            if not os.path.exists(log_file):
                return []
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse log entries (format: [YYYY-MM-DD HH:MM:SS] Action)
            activities = []
            for line in reversed(lines[-max_entries:]):  # Get last N entries, newest first
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Extract timestamp and action
                    if line.startswith('[') and ']' in line:
                        timestamp_end = line.index(']')
                        timestamp = line[1:timestamp_end]
                        action = line[timestamp_end+2:].strip()
                        
                        # Categorize action for color coding
                        action_type = 'info'
                        if 'Added' in action or 'Created' in action:
                            action_type = 'success'
                        elif 'Deleted' in action:
                            action_type = 'danger'
                        elif 'Updated' in action or 'Restored' in action:
                            action_type = 'primary'
                        elif 'logged in' in action or 'logged out' in action:
                            action_type = 'secondary'
                        
                        activities.append({
                            'timestamp': timestamp,
                            'action': action,
                            'type': action_type
                        })
                except:
                    continue
            
            return activities
        except Exception as e:
            print(f"Error reading activity log: {e}")
            return []
    
    # ============ PROJECT OPERATIONS ============
    
    def add_project(self, project_data, id_prefix='', content_string=None):
        """Add new project or research to data.json. Returns new ID if success, None otherwise."""
        data = self.read_data()  # This is a list
        
        # Generate ID
        existing_ids = [item.get('id', '') for item in data]
        
        # If prefix is provided (e.g. 'paper_'), use it in ID generation
        candidate_name = id_prefix + project_data['title'] if id_prefix else project_data['title']
        project_data['id'] = self._generate_id(candidate_name, existing_ids)
        
        # Handle content string if provided
        if content_string:
            filename = self.save_content_string(project_data['id'], content_string)
            if filename:
                project_data['content_file'] = filename

        data.append(project_data)
        if self.write_data(data):
            return project_data['id']
        return None
    
    def update_project(self, project_id, project_data):
        """Update existing project"""
        data = self.read_data()  # This is a list
        
        for i, item in enumerate(data):
            if item.get('id') == project_id:
                # Preserve ID
                project_data['id'] = project_id
                data[i] = project_data
                return self.write_data(data)
        
        return False
    
    def delete_project(self, project_id):
        """Delete project"""
        data = self.read_data()  # This is a list
        data = [item for item in data if item.get('id') != project_id]
        return self.write_data(data)
    
    def get_project(self, project_id):
        """Get single project by ID"""
        data = self.read_data()  # This is a list
        for item in data:
            if item.get('id') == project_id:
                return item
        return None
    
    # ============ IMAGE OPERATIONS ============
    
    def upload_image(self, file):
        """Upload image to static folder"""
        if not file:
            return None
        
        filename = secure_filename(file.filename)
        
        # Add timestamp to avoid conflicts
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(self.static_dir, filename)
        
        try:
            file.save(filepath)
            return filename
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None
    
    def delete_image(self, filename):
        """Delete image from static folder"""
        filepath = os.path.join(self.static_dir, filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
    
    def list_images(self):
        """List all images in static folder with metadata and usage tracking"""
        try:
            image_files = [f for f in os.listdir(self.static_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
            
            # Get all projects/research to check usage
            data = self.read_data()
            
            images_info = []
            for filename in sorted(image_files):
                filepath = os.path.join(self.static_dir, filename)
                
                # Check which items use this image
                used_by = []
                for item in data:
                    if item.get('image') == filename:
                        used_by.append(item.get('title', item.get('id', 'Unknown')))
                
                images_info.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'used_by': used_by,
                    'is_orphaned': len(used_by) == 0
                })
            
            return images_info
        except Exception as e:
            print(f"Error listing images: {e}")
            return []
    
    # ============ CONTENT FILE OPERATIONS ============
    
    def upload_content_file(self, file, project_id):
        """Upload markdown or html content file"""
        if not file:
            return None
            
        # Check extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.md', '.html', '.htm']:
            return None
            
        # Create filename: project_id.ext
        # This ensures one content file per project per type (or overwrite)
        # To avoid multiple files (id.md and id.html), we might want to delete siblings?
        # For now, just save it.
        filename = secure_filename(f"{project_id}{ext}")
        content_dir = os.path.join(self.base_dir, 'templates', 'content')
        os.makedirs(content_dir, exist_ok=True)
        
        filepath = os.path.join(content_dir, filename)
        
        try:
            file.save(filepath)
            return filename
        except Exception as e:
            print(f"Error uploading content: {e}")
            return None

    def save_content_string(self, project_id, content_string):
        """Save string content as markdown file"""
        if not content_string:
            return None
            
        filename = f"{project_id}.md"
        content_dir = os.path.join(self.base_dir, 'templates', 'content')
        os.makedirs(content_dir, exist_ok=True)
        
        filepath = os.path.join(content_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content_string)
            return filename
        except Exception as e:
            print(f"Error saving content string: {e}")
            return None

    def upload_content_asset(self, file_obj):
        """Upload a generic asset file (image) for use in content"""
        if not file_obj:
            return None
        
        # Secure filename
        original_name = secure_filename(file_obj.filename)
        # Add timestamp to prevent overwrites
        base, ext = os.path.splitext(original_name)
        filename = f"{base}_{int(time.time())}{ext}"
         
        # Ensure static/uploads/content exists
        upload_dir = os.path.join(self.base_dir, 'static', 'uploads', 'content')
        os.makedirs(upload_dir, exist_ok=True)
        
        filepath = os.path.join(upload_dir, filename)
        
        try:
            file_obj.save(filepath)
            # Return relative path for use in <img> src
            return f"uploads/content/{filename}"
        except Exception as e:
            print(f"Error uploading asset: {e}")
            return None

    def delete_content_file(self, filename):
        """Delete content file"""
        if not filename: return False
        
        filepath = os.path.join(self.base_dir, 'templates', 'content', filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error deleting content: {e}")
            return False

    # ============ HELPER METHODS ============
    
    def _generate_id(self, title, existing_ids):
        """Generate unique ID from title"""
        base_id = title.lower().replace(' ', '_').replace('-', '_')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')
        
        # Ensure uniqueness
        id_candidate = base_id
        counter = 1
        while id_candidate in existing_ids:
            id_candidate = f"{base_id}_{counter}"
            counter += 1
        
        return id_candidate
