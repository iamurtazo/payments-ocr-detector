"""
Batch processor for analyzing multiple payment screenshots
"""

import os
import json
from typing import List, Dict
from payment_screenshot_detector import PaymentScreenshotDetector, PaymentInfo
import argparse
from pathlib import Path

class BatchPaymentAnalyzer:
    def __init__(self):
        self.detector = PaymentScreenshotDetector()
        
    def process_directory(self, directory_path: str, output_file: str = None) -> Dict:
        """Process all images in a directory."""
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        results = []
        summary = {
            'total_images': 0,
            'payment_screenshots': 0,
            'successful_payments': 0,
            'failed_payments': 0,
            'recommended_deletions': 0,
            'processing_errors': 0
        }
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(directory_path).glob(f"*{ext}"))
            image_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} image files to process...")
        
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"Processing {i}/{len(image_files)}: {image_path.name}")
                
                # Analyze the image
                payment_info = self.detector.analyze_payment_screenshot(str(image_path))
                should_delete = self.detector.should_delete_screenshot(payment_info)
                
                # Store result
                result = {
                    'file_name': image_path.name,
                    'file_path': str(image_path),
                    'status': payment_info.status.value,
                    'confidence_score': payment_info.confidence_score,
                    'currency': payment_info.currency,
                    'amount': payment_info.amount,
                    'payment_method': payment_info.payment_method,
                    'transaction_id': payment_info.transaction_id,
                    'detected_phrases': payment_info.detected_phrases or [],
                    'recommend_deletion': should_delete
                }
                
                results.append(result)
                
                # Update summary
                summary['total_images'] += 1
                
                if payment_info.status.value != 'not_payment_screenshot':
                    summary['payment_screenshots'] += 1
                
                if payment_info.status.value == 'payment_successful':
                    summary['successful_payments'] += 1
                elif payment_info.status.value == 'payment_failed':
                    summary['failed_payments'] += 1
                
                if should_delete:
                    summary['recommended_deletions'] += 1
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                summary['processing_errors'] += 1
        
        # Prepare final results
        final_results = {
            'summary': summary,
            'results': results
        }
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return final_results
    
    def generate_deletion_list(self, results: Dict, output_file: str = None) -> List[str]:
        """Generate a list of files recommended for deletion."""
        
        deletion_candidates = []
        
        for result in results['results']:
            if result['recommend_deletion']:
                deletion_candidates.append(result['file_path'])
        
        if output_file:
            with open(output_file, 'w') as f:
                for file_path in deletion_candidates:
                    f.write(f"{file_path}\n")
            print(f"Deletion list saved to: {output_file}")
        
        return deletion_candidates

def main():
    parser = argparse.ArgumentParser(description='Batch process payment screenshots')
    parser.add_argument('directory', help='Directory containing screenshot images')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--deletion-list', '-d', help='Output file for deletion candidates')
    
    args = parser.parse_args()
    
    # Initialize batch analyzer
    analyzer = BatchPaymentAnalyzer()
    
    # Process directory
    results = analyzer.process_directory(args.directory, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images processed: {results['summary']['total_images']}")
    print(f"Payment screenshots detected: {results['summary']['payment_screenshots']}")
    print(f"Successful payments: {results['summary']['successful_payments']}")
    print(f"Failed payments: {results['summary']['failed_payments']}")
    print(f"Recommended for deletion: {results['summary']['recommended_deletions']}")
    print(f"Processing errors: {results['summary']['processing_errors']}")
    
    # Generate deletion list if requested
    if args.deletion_list:
        deletion_candidates = analyzer.generate_deletion_list(results, args.deletion_list)
        print(f"\nGenerated deletion list with {len(deletion_candidates)} files")

if __name__ == "__main__":
    main()