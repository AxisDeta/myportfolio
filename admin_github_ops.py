"""
GitHub Operations Module
Handles GitHub API operations for portfolio management in production
Mirrors the interface of admin_file_ops.py but uses GitHub API instead of local files
"""

import json
import os
import base64
from datetime import datetime
from github import Github, Auth, GithubException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AdminGitHubOps:
    def __init__(self):
        """Initialize GitHub API connection"""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        
        self.repo_name = os.getenv('GITHUB_REPO', 'SHIVOGOJOHN/Profile')
        self.branch = os.getenv('GITHUB_BRANCH', 'main')
        
        # Initialize GitHub client
        auth = Auth.Token(token)
        self.github = Github(auth=auth)
        self.repo = self.github.get_repo(self.repo_name)
        
        print(f"[GitHub] Connected to {self.repo_name} on branch {self.branch}")
    
    def _get_file_sha(self, path):
        """Get the SHA of a file in the repository (needed for updates)"""
        try:
            contents = self.repo.get_contents(path, ref=self.branch)
            return contents.sha
        except GithubException:
            return None  # File doesn't exist
    
    def _commit_file(self, path, content, message, is_binary=False):
        """Create or update a file in the repository"""
        try:
            sha = self._get_file_sha(path)
            
            # Encode content
            if is_binary:
                # Content is already bytes
                encoded_content = base64.b64encode(content).decode('utf-8')
            else:
                # Content is string, encode to bytes first
                if isinstance(content, str):
                    content = content.encode('utf-8')
                encoded_content = base64.b64encode(content).decode('utf-8')
            
            if sha:
                # Update existing file
                self.repo.update_file(
                    path=path,
                    message=message,
                    content=encoded_content,
                    sha=sha,
                    branch=self.branch
                )
            else:
                # Create new file
                self.repo.create_file(
                    path=path,
                    message=message,
                    content=encoded_content,
                    branch=self.branch
                )
            
            print(f"[GitHub] {message}")
            return True
        except Exception as e:
            print(f"[GitHub] Error committing {path}: {e}")
            return False
    
    def _delete_file(self, path, message):
        """Delete a file from the repository"""
        try:
            sha = self._get_file_sha(path)
            if sha:
                self.repo.delete_file(
                    path=path,
                    message=message,
                    sha=sha,
                    branch=self.branch
                )
                print(f"[GitHub] {message}")
                return True
            return False
        except Exception as e:
            print(f"[GitHub] Error deleting {path}: {e}")
            return False
    
    def _read_file(self, path):
        """Read a file from the repository"""
        try:
            contents = self.repo.get_contents(path, ref=self.branch)
            return contents.decoded_content.decode('utf-8')
        except Exception as e:
            print(f"[GitHub] Error reading {path}: {e}")
            return None
    
    # ============ DATA OPERATIONS ============
    
    def read_settings(self):
        """Read settings.json from GitHub"""
        try:
            content = self._read_file('settings.json')
            if content:
                return json.loads(content)
            return {}
        except Exception as e:
            print(f"Error reading settings: {e}")
            return {}
    
    def write_settings(self, settings):
        """Write settings.json to GitHub"""
        try:
            content = json.dumps(settings, indent=2, ensure_ascii=False)
            return self._commit_file(
                'settings.json',
                content,
                '[Admin] Update: Hero metrics'
            )
        except Exception as e:
            print(f"Error writing settings: {e}")
            return False
    
    def read_data(self):
        """Read data.json from GitHub"""
        try:
            content = self._read_file('data.json')
            if content:
                return json.loads(content)
            return []
        except Exception as e:
            print(f"Error reading data: {e}")
            return []
    
    def write_data(self, data):
        """Write data.json to GitHub"""
        try:
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return self._commit_file(
                'data.json',
                content,
                '[Admin] Update: Project/Research data'
            )
        except Exception as e:
            print(f"Error writing data: {e}")
            return False
    
    # ============ BACKUP OPERATIONS ============
    
    def backup_data(self):
        """Create backup of data.json in GitHub"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f'backups/data_backup_{timestamp}.json'
            
            # Read current data
            data = self.read_data()
            content = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Commit backup
            success = self._commit_file(
                backup_path,
                content,
                f'[Admin] Backup: Created data backup {timestamp}'
            )
            
            return backup_path if success else None
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    def restore_data(self, backup_filename):
        """Restore data.json from a backup in GitHub"""
        try:
            backup_path = f'backups/{backup_filename}'
            content = self._read_file(backup_path)
            
            if content:
                # Validate JSON
                data = json.loads(content)
                # Write to data.json
                return self._commit_file(
                    'data.json',
                    content,
                    f'[Admin] Restore: From backup {backup_filename}'
                )
            return False
        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self):
        """List all backups in GitHub repository"""
        try:
            contents = self.repo.get_contents('backups', ref=self.branch)
            backups_info = []
            
            for item in contents:
                if item.name.startswith('data_backup_') and item.name.endswith('.json'):
                    # Parse timestamp
                    try:
                        timestamp_str = item.name.replace('data_backup_', '').replace('.json', '')
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Read backup to get summary
                        backup_content = self._read_file(f'backups/{item.name}')
                        if backup_content:
                            data = json.loads(backup_content)
                            projects = [i for i in data if not i.get('id', '').startswith('paper_')]
                            research = [i for i in data if i.get('id', '').startswith('paper_')]
                            summary = f"{len(projects)} projects, {len(research)} research items"
                        else:
                            summary = "Unable to read backup"
                        
                        backups_info.append({
                            'filename': item.name,
                            'timestamp': formatted_time,
                            'summary': summary,
                            'size': item.size
                        })
                    except:
                        backups_info.append({
                            'filename': item.name,
                            'timestamp': 'Unknown',
                            'summary': 'Unable to parse',
                            'size': item.size
                        })
            
            # Sort by filename (which includes timestamp) in reverse
            backups_info.sort(key=lambda x: x['filename'], reverse=True)
            return backups_info
        except Exception as e:
            print(f"Error listing backups: {e}")
            return []
    
    def delete_backup(self, filename):
        """Delete a backup file from GitHub"""
        # Security check
        if not filename.startswith('data_backup_') or not filename.endswith('.json'):
            return False
        if '/' in filename or '\\' in filename or '..' in filename:
            return False
        
        return self._delete_file(
            f'backups/{filename}',
            f'[Admin] Delete: Backup {filename}'
        )
    
    def read_activity_log(self, max_entries=30):
        """Read activity log from GitHub"""
        try:
            content = self._read_file('admin_log.txt')
            if not content:
                return []
            
            lines = content.strip().split('\n')
            activities = []
            
            for line in reversed(lines[-max_entries:]):
                if not line.strip():
                    continue
                
                try:
                    if line.startswith('[') and ']' in line:
                        timestamp_end = line.index(']')
                        timestamp = line[1:timestamp_end]
                        action = line[timestamp_end+2:].strip()
                        
                        # Categorize action
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
        """Add new project/research to data.json in GitHub"""
        try:
            data = self.read_data()
            
            # Generate ID
            existing_ids = [item.get('id', '') for item in data]
            new_id = self._generate_id(project_data.get('title', 'Untitled'), existing_ids)
            
            if id_prefix:
                new_id = f"{id_prefix}{new_id}"
            
            project_data['id'] = new_id
            data.append(project_data)
            
            # Save data
            if self.write_data(data):
                # Save content if provided
                if content_string:
                    self.save_content_string(new_id, content_string)
                return new_id
            return None
        except Exception as e:
            print(f"Error adding project: {e}")
            return None
    
    def update_project(self, project_id, project_data):
        """Update existing project in GitHub"""
        try:
            data = self.read_data()
            
            for i, item in enumerate(data):
                if item.get('id') == project_id:
                    # Preserve ID
                    project_data['id'] = project_id
                    data[i] = project_data
                    return self.write_data(data)
            
            return False
        except Exception as e:
            print(f"Error updating project: {e}")
            return False
    
    def delete_project(self, project_id):
        """Delete project from GitHub"""
        try:
            data = self.read_data()
            data = [item for item in data if item.get('id') != project_id]
            return self.write_data(data)
        except Exception as e:
            print(f"Error deleting project: {e}")
            return False
    
    def get_project(self, project_id):
        """Get single project by ID from GitHub"""
        data = self.read_data()
        for item in data:
            if item.get('id') == project_id:
                return item
        return None
    
    # ============ IMAGE OPERATIONS ============
    
    def upload_image(self, file_obj):
        """Upload image to static/ folder in GitHub"""
        try:
            from werkzeug.utils import secure_filename
            import time
            
            filename = secure_filename(file_obj.filename)
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{timestamp}{ext}"
            
            # Read file content
            file_content = file_obj.read()
            
            # Commit to GitHub
            success = self._commit_file(
                f'static/{new_filename}',
                file_content,
                f'[Admin] Upload: Image {new_filename}',
                is_binary=True
            )
            
            return new_filename if success else None
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None
    
    def delete_image(self, filename):
        """Delete image from static/ folder in GitHub"""
        try:
            from werkzeug.utils import secure_filename
            
            filename = secure_filename(filename)
            return self._delete_file(
                f'static/{filename}',
                f'[Admin] Delete: Image {filename}'
            )
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
    
    def list_images(self):
        """List all images in static/ folder with usage tracking"""
        try:
            contents = self.repo.get_contents('static', ref=self.branch)
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
            
            # Get all projects/research to check usage
            data = self.read_data()
            
            images_info = []
            for item in contents:
                if item.name.lower().endswith(image_extensions):
                    # Check usage
                    used_by = []
                    for proj in data:
                        if proj.get('image') == item.name:
                            used_by.append(proj.get('title', proj.get('id', 'Unknown')))
                    
                    images_info.append({
                        'filename': item.name,
                        'size': item.size,
                        'used_by': used_by,
                        'is_orphaned': len(used_by) == 0
                    })
            
            images_info.sort(key=lambda x: x['filename'])
            return images_info
        except Exception as e:
            print(f"Error listing images: {e}")
            return []
    
    # ============ CONTENT FILE OPERATIONS ============
    
    def save_content_string(self, project_id, content_string):
        """Save markdown content to templates/content/ in GitHub"""
        try:
            filename = f"{project_id}.md"
            path = f'templates/content/{filename}'
            
            return self._commit_file(
                path,
                content_string,
                f'[Admin] Update: Content for {project_id}'
            )
        except Exception as e:
            print(f"Error saving content: {e}")
            return False
    
    def upload_content_asset(self, file_obj):
        """Upload asset (image) for content to static/uploads/content/ in GitHub"""
        try:
            from werkzeug.utils import secure_filename
            import time
            
            filename = secure_filename(file_obj.filename)
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{timestamp}{ext}"
            
            # Read file content
            file_content = file_obj.read()
            
            # Commit to GitHub
            success = self._commit_file(
                f'static/uploads/content/{new_filename}',
                file_content,
                f'[Admin] Upload: Content asset {new_filename}',
                is_binary=True
            )
            
            if success:
                return f'uploads/content/{new_filename}'
            return None
        except Exception as e:
            print(f"Error uploading content asset: {e}")
            return None
    
    def delete_content_file(self, filename):
        """Delete content file from templates/content/ in GitHub"""
        try:
            from werkzeug.utils import secure_filename
            
            filename = secure_filename(filename)
            return self._delete_file(
                f'templates/content/{filename}',
                f'[Admin] Delete: Content file {filename}'
            )
        except Exception as e:
            print(f"Error deleting content file: {e}")
            return False
    
    # ============ HELPER METHODS ============
    
    def _generate_id(self, title, existing_ids):
        """Generate unique ID from title"""
        import re
        
        # Convert to lowercase and replace spaces/special chars with underscores
        base_id = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        
        # Ensure uniqueness
        unique_id = base_id
        counter = 1
        while unique_id in existing_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        
        return unique_id
