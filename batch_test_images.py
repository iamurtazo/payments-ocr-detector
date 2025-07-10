#!/usr/bin/env python3
"""
Batch test script for payment screenshot detector
Tests all images in "Images payments" folder and generates CSV results
"""

import os
import csv
import glob
from pathlib import Path
from payment_screenshot_detector import PaymentScreenshotDetector

def process_images_batch():
    """Process all images in the Images payments folder and generate CSV results."""
    
    # Initialize the detector
    detector = PaymentScreenshotDetector()
    
    # Get all image files from the Images payments folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(f"Images payments/{ext}"))
        image_files.extend(glob.glob(f"Images payments/{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process...")
    
    # Prepare CSV data
    csv_data = []
    
    # Add the existing test results first
    existing_results = [
        {
            'Image': 'images/tossbank3.jpg',
            'Status': 'payment_successful',
            'Confidence_Score': 0.85,
            'Payment_Score': 8,
            'Original_Text': 'OOH QE 수 31000 원 을 보 냈 어이 Lom | @',
            'Normalized_Text': 'OOH QE 수 31000 원을보냈어이 Lom | @',
            'Detected_Phrases': '원, High payment score: 8',
            'Recommended_Deletion': 'YES'
        }
    ]
    csv_data.extend(existing_results)
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Analyze the image
            result = detector.analyze_payment_screenshot(image_path)
            
            # Extract additional information for CSV
            ocr_results = detector.extract_text_with_ocr(image_path)
            full_text = " ".join([result.text for result in ocr_results]) if ocr_results else ""
            normalized_text = detector.normalize_korean_text(full_text)
            payment_score = detector.calculate_payment_score(full_text)
            should_delete = detector.should_delete_screenshot(result)
            
            # Prepare CSV row
            csv_row = {
                'Image': image_path,
                'Status': result.status.value,
                'Confidence_Score': round(result.confidence_score, 2),
                'Payment_Score': payment_score,
                'Original_Text': full_text[:200] + "..." if len(full_text) > 200 else full_text,
                'Normalized_Text': normalized_text[:200] + "..." if len(normalized_text) > 200 else normalized_text,
                'Detected_Phrases': ', '.join(result.detected_phrases) if result.detected_phrases else '',
                'Recommended_Deletion': 'YES' if should_delete else 'NO'
            }
            
            csv_data.append(csv_row)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Add error row to CSV
            csv_row = {
                'Image': image_path,
                'Status': 'ERROR',
                'Confidence_Score': 0.0,
                'Payment_Score': 0,
                'Original_Text': f'Error: {str(e)}',
                'Normalized_Text': '',
                'Detected_Phrases': '',
                'Recommended_Deletion': 'NO'
            }
            csv_data.append(csv_row)
    
    # Write results to CSV
    csv_filename = 'results.csv'
    fieldnames = ['Image', 'Status', 'Confidence_Score', 'Payment_Score', 'Original_Text', 'Normalized_Text', 'Detected_Phrases', 'Recommended_Deletion']
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nResults saved to: {csv_filename}")
    
    # Print summary
    total_images = len(csv_data)
    payment_screenshots = sum(1 for row in csv_data if 'payment_successful' in row['Status'])
    recommended_deletions = sum(1 for row in csv_data if row['Recommended_Deletion'] == 'YES')
    errors = sum(1 for row in csv_data if row['Status'] == 'ERROR')
    
    print(f"\nSummary:")
    print(f"Total images processed: {total_images}")
    print(f"Payment screenshots detected: {payment_screenshots}")
    print(f"Recommended for deletion: {recommended_deletions}")
    print(f"Processing errors: {errors}")

if __name__ == "__main__":
    process_images_batch() 