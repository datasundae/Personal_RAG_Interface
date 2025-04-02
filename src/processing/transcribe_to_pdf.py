import whisper
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from pathlib import Path
import torch
from tqdm import tqdm
import time
import multiprocessing
import argparse
import textwrap
import re
import soundfile as sf
import numpy as np

def transcribe_audio(audio_path, output_dir, progress_callback=None):
    """Transcribe audio file using Whisper"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with progress bar
    if progress_callback:
        progress_callback("Loading Whisper model...")
    # Use tiny model for faster transcription
    model = whisper.load_model("tiny", device=device)
    
    # Transcribe with progress bar
    if progress_callback:
        progress_callback("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,  # Use FP32 for CPU
        beam_size=1,  # Reduce beam size for faster processing
        best_of=1,  # Reduce number of candidates
        temperature=0.0,  # Use greedy decoding for faster processing
        condition_on_previous_text=False,  # Disable conditioning for faster processing
        no_speech_threshold=0.6,  # Adjust threshold for faster processing
        logprob_threshold=-1.0,  # Disable logprob threshold
        compression_ratio_threshold=1.2  # Adjust compression ratio threshold
    )
    
    # Save transcription
    output_file = os.path.join(output_dir, "transcription.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    return output_file

def create_pdf(text_file, output_dir, progress_callback=None):
    """Create PDF from transcription text"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read transcription
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Normalize text
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Create PDF
    output_file = os.path.join(output_dir, "transcription.pdf")
    c = canvas.Canvas(output_file, pagesize=A4)
    
    # Set margins (20mm on all sides)
    width, height = A4
    margin = 20 * 2.83465  # Convert mm to points (1mm = 2.83465 points)
    c.setPageSize((width, height))
    
    # Set font and size
    c.setFont("Helvetica", 12)
    
    # Calculate text width and height
    text_width = width - 2 * margin
    line_height = 14  # Approximate line height in points
    
    # Process text in chunks
    paragraphs = text.split('\n\n')
    y = height - margin - line_height
    
    if progress_callback:
        progress_callback("Adding text to PDF...")
    
    for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
        # Split paragraph into words
        words = paragraph.split()
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = c.stringWidth(word + ' ', "Helvetica", 12)
            if current_width + word_width <= text_width:
                current_line.append(word)
                current_width += word_width
            else:
                # Write current line
                c.drawString(margin, y, ' '.join(current_line))
                y -= line_height
                
                # Check if we need a new page
                if y < margin + line_height:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = height - margin - line_height
                
                current_line = [word]
                current_width = word_width
        
        # Write remaining words
        if current_line:
            c.drawString(margin, y, ' '.join(current_line))
            y -= line_height
        
        # Add paragraph spacing
        y -= line_height
        
        # Check if we need a new page
        if y < margin + line_height:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - margin - line_height
    
    c.save()
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to text and create PDF")
    parser.add_argument("--mode", choices=["transcribe", "pdf", "both"], default="both",
                      help="Operation mode: transcribe (audio to text), pdf (text to PDF), or both")
    parser.add_argument("--audio", required=True,
                      help="Path to input audio file")
    parser.add_argument("--output-dir", required=True,
                      help="Path to output directory")
    parser.add_argument("--output-name", default="transcription",
                      help="Base name for output files (without extension)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ["transcribe", "both"]:
        print("Transcribing audio...")
        text_file = transcribe_audio(args.audio, args.output_dir)
        # Rename the file to use the specified output name
        new_text_file = os.path.join(args.output_dir, f"{args.output_name}.txt")
        os.rename(text_file, new_text_file)
        print(f"Transcription saved to: {new_text_file}")
    
    if args.mode in ["pdf", "both"]:
        print("Creating PDF...")
        text_file = os.path.join(args.output_dir, f"{args.output_name}.txt")
        if not os.path.exists(text_file):
            print("Error: Transcription file not found. Please run transcription first.")
            return
        
        pdf_file = create_pdf(text_file, args.output_dir)
        # Rename the PDF file to use the specified output name
        new_pdf_file = os.path.join(args.output_dir, f"{args.output_name}.pdf")
        os.rename(pdf_file, new_pdf_file)
        print(f"PDF saved to: {new_pdf_file}")

if __name__ == "__main__":
    main() 