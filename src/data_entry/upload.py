import os
from typing import Dict, Any
from werkzeug.utils import secure_filename
import logging

class DocumentUploader:
    def __init__(self, upload_folder: str, allowed_extensions: set):
        self.upload_folder = upload_folder
        self.allowed_extensions = allowed_extensions
        self.logger = logging.getLogger(__name__)
        
        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
    def allowed_file(self, filename: str) -> bool:
        """Check if the file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
               
    def save_file(self, file) -> Dict[str, Any]:
        """Save the uploaded file and return its metadata"""
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.upload_folder, filename)
            file.save(file_path)
            
            return {
                'filename': filename,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path)
            }
        return None
        
    def process_upload(self, file, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process an uploaded file and return its information"""
        file_info = self.save_file(file)
        if file_info:
            if metadata:
                file_info.update(metadata)
            self.logger.info(f"Successfully processed upload: {file_info}")
            return file_info
        return None 