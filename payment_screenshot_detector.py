import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import argparse
import os
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

class PaymentStatus(Enum):
    PAYMENT_SUCCESSFUL = "payment_successful"
    TRANSFER_COMPLETE = "transfer_complete"
    PAYMENT_FAILED = "payment_failed"
    NOT_PAYMENT_SCREENSHOT = "not_payment_screenshot"
    UNKNOWN = "unknown"

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]

@dataclass
class PaymentInfo:
    status: PaymentStatus
    currency: Optional[str] = None
    amount: Optional[str] = None
    recipient: Optional[str] = None
    payment_method: Optional[str] = None
    transaction_id: Optional[str] = None
    detected_phrases: List[str] = None
    confidence_score: float = 0.0

class PaymentScreenshotDetector:
    def __init__(self):
        """Initialize the payment screenshot detector with predefined patterns."""
        
        # Success patterns - case insensitive (English)
        self.success_patterns = [
            r'payment\s*successful',
            r'transfer\s*complete',
            r'transaction\s*successful',
            r'payment\s*sent',
            r'transfer\s*sent',
            r'money\s*sent',
            r'payment\s*confirmed',
            r'transfer\s*confirmed',
            r'successfully\s*sent',
            r'payment\s*completed',
            r'transfer\s*completed',
            r'transaction\s*completed',
            r'sent\s*successfully',
            r'payment\s*processed',
            r'transfer\s*processed',
            r'completed',
            r'successful',
            r'confirmed',
            r'sent'
        ]
        
        # Korean success patterns
        self.korean_success_patterns = [
            r'완료',  # Complete
            r'성공',  # Success
            r'전송완료',  # Transfer complete
            r'송금완료',  # Remittance complete
            r'이체완료',  # Transfer complete
            r'처리완료',  # Processing complete
            r'승인',  # Approval
            r'확인',  # Confirmation
            r'완료되었습니다',  # Has been completed
            r'성공적으로',  # Successfully
            r'전송되었습니다',  # Has been sent
            r'송금되었습니다',  # Has been remitted
            r'이체되었습니다',  # Has been transferred
            r'처리되었습니다',  # Has been processed
            r'승인되었습니다',  # Has been approved
            r'확인되었습니다',  # Has been confirmed
            # Toss Bank specific patterns
            r'보냈어요',  # "has been sent" - very common in Toss
            r'보냈습니다',  # "has been sent" (formal)
            r'송금되었습니다',  # "transfer has been completed"
            r'이체되었습니다',  # "transfer has been completed"
            r'결제완료',  # "payment completed"
            r'송금완료',  # "transfer completed"
            r'이체완료',  # "transfer completed"
            r'처리완료',  # "processing completed"
            r'승인완료',  # "approval completed"
            r'확인완료'  # "confirmation completed"
        ]
        
        # Failure patterns (English)
        self.failure_patterns = [
            r'payment\s*failed',
            r'transfer\s*failed',
            r'transaction\s*failed',
            r'payment\s*declined',
            r'insufficient\s*funds',
            r'payment\s*error',
            r'transfer\s*error',
            r'transaction\s*error',
            r'failed',
            r'error',
            r'declined',
            r'rejected'
        ]
        
        # Korean failure patterns
        self.korean_failure_patterns = [
            r'실패',  # Failed
            r'오류',  # Error
            r'거절',  # Rejected
            r'취소',  # Cancelled
            r'실패했습니다',  # Has failed
            r'오류가 발생했습니다',  # An error occurred
            r'거절되었습니다',  # Has been rejected
            r'취소되었습니다',  # Has been cancelled
            r'잔액부족',  # Insufficient balance
            r'한도초과',  # Limit exceeded
            r'계좌번호 오류',  # Account number error
            r'수취인 정보 오류'  # Recipient information error
        ]
        
        # Currency patterns with symbols and codes
        self.currency_patterns = [
            r'[\$][\d,]+\.?\d*',  # USD with $ symbol
            r'[₩][\d,]+',         # Korean Won
            r'[€][\d,]+\.?\d*',   # Euro
            r'[£][\d,]+\.?\d*',   # British Pound
            r'[₽][\d,]+\.?\d*',   # Russian Ruble
            r'[¥][\d,]+\.?\d*',   # Japanese Yen/Chinese Yuan
            r'[₹][\d,]+\.?\d*',   # Indian Rupee
            r'\bUSD\s*[\d,]+\.?\d*',
            r'\bEUR\s*[\d,]+\.?\d*',
            r'\bGBP\s*[\d,]+\.?\d*',
            r'\bKRW\s*[\d,]+',
            r'\bJPY\s*[\d,]+',
            r'\bINR\s*[\d,]+\.?\d*',
            r'\bRUB\s*[\d,]+\.?\d*',
            # Korean amount patterns (Korean Won amounts)
            r'[\d,]+원',  # Amount in Won (e.g., 10,000원)
            r'[\d,]+\.?\d*원',  # Amount with decimal in Won
            r'[\d,]+,\d{3}원',  # Amount with comma formatting in Won
            r'[\d,]+\.?\d*만원',  # Amount in 10,000 Won units
            r'[\d,]+\.?\d*천원',  # Amount in 1,000 Won units
            r'[\d,]+\.?\d*억원'   # Amount in 100 million Won units
        ]
        
        # Payment method patterns (English)
        self.payment_method_patterns = [
            r'visa\s*\*+\d{4}',
            r'mastercard\s*\*+\d{4}',
            r'paypal',
            r'apple\s*pay',
            r'google\s*pay',
            r'venmo',
            r'zelle',
            r'cash\s*app',
            r'bank\s*transfer',
            r'wire\s*transfer',
            r'credit\s*card',
            r'debit\s*card'
        ]
        
        # Korean payment method patterns
        self.korean_payment_method_patterns = [
            r'하나은행',  # Hana Bank
            r'농협은행',  # NongHyup Bank
            r'토스',  # Toss
            r'카카오뱅크',  # Kakao Bank
            r'신한은행',  # Shinhan Bank
            r'국민은행',  # KB Bank
            r'우리은행',  # Woori Bank
            r'기업은행',  # IBK Bank
            r'새마을금고',  # Saemaul Bank
            r'신용카드',  # Credit card
            r'체크카드',  # Debit card
            r'계좌이체',  # Account transfer
            r'실시간이체',  # Real-time transfer
            r'자동이체',  # Automatic transfer
            r'스마트폰뱅킹',  # Smartphone banking
            r'인터넷뱅킹',  # Internet banking
            r'모바일뱅킹'  # Mobile banking
        ]
        
        # Transaction ID patterns (English)
        self.transaction_id_patterns = [
            r'transaction\s*id\s*:?\s*([a-zA-Z0-9]+)',
            r'reference\s*:?\s*([a-zA-Z0-9]+)',
            r'confirmation\s*:?\s*([a-zA-Z0-9]+)',
            r'order\s*id\s*:?\s*([a-zA-Z0-9]+)',
            r'txn\s*id\s*:?\s*([a-zA-Z0-9]+)'
        ]
        
        # Korean transaction ID patterns
        self.korean_transaction_id_patterns = [
            r'거래번호\s*:?\s*([a-zA-Z0-9]+)',
            r'참조번호\s*:?\s*([a-zA-Z0-9]+)',
            r'승인번호\s*:?\s*([a-zA-Z0-9]+)',
            r'주문번호\s*:?\s*([a-zA-Z0-9]+)',
            r'거래일시\s*:?\s*([0-9]{14})',  # YYYYMMDDHHMMSS format
            r'승인일시\s*:?\s*([0-9]{14})',
            r'처리일시\s*:?\s*([0-9]{14})'
        ]
        
        # Korean bank-specific keywords
        self.korean_bank_keywords = [
            r'은행',  # Bank
            r'계좌',  # Account
            r'송금',  # Remittance
            r'이체',  # Transfer
            r'입금',  # Deposit
            r'출금',  # Withdrawal
            r'잔액',  # Balance
            r'수수료',  # Fee
            r'수취인',  # Recipient
            r'발신인',  # Sender
            r'계좌번호',  # Account number
            r'예금주',  # Account holder
            r'거래내역',  # Transaction history
            r'승인',  # Approval
            r'확인',  # Confirmation
            r'완료',  # Complete
            r'처리',  # Processing
            r'성공',  # Success
            r'실패',  # Failed
            r'오류',  # Error
            r'취소',  # Cancel
            r'거절',  # Reject
            # Toss Bank specific UI elements
            r'공유하기',  # Share button
            r'메모 남기기',  # Leave a memo
            r'확인',  # Confirm button
            r'취소',  # Cancel button
            r'뒤로',  # Back button
            r'홈',  # Home button
            r'보내기',  # Send
            r'받기',  # Receive
            r'송금',  # Transfer
            r'이체',  # Transfer
            r'결제',  # Payment
            r'원',  # Won currency
            r'만원',  # 10,000 Won
            r'천원',  # 1,000 Won
            r'억원'  # 100 million Won
        ]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def extract_text_with_ocr(self, image_path: str) -> List[OCRResult]:
        """Extract text from image using Tesseract OCR with Korean language support."""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return []
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Configure Tesseract for Korean + English
            # Try Korean first, then fallback to English
            custom_config = r'--oem 3 --psm 6 -l kor+eng'
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(
                pil_image, 
                config=custom_config, 
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 20:  # Lower threshold for Korean text
                    text = data['text'][i].strip()
                    if text:  # Only include non-empty text
                        bbox = (
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                        results.append(OCRResult(
                            text=text,
                            confidence=int(data['conf'][i]),
                            bounding_box=bbox
                        ))
            
            return results
        
        except Exception as e:
            print(f"OCR extraction error: {str(e)}")
            return []

    def extract_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Extract matching patterns from text."""
        matches = []
        text_lower = text.lower()
        
        for pattern in patterns:
            found = re.findall(pattern, text_lower, re.IGNORECASE)
            matches.extend(found)
        
        return matches

    def extract_korean_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Extract matching Korean patterns from text."""
        matches = []
        
        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
        
        return matches

    def detect_currency_and_amount(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect currency and amount from text with Korean Won support."""
        for pattern in self.currency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_text = match.group(0)
                
                # Extract currency symbol/code and amount
                currency_symbols = {
                    '$': 'USD', '₩': 'KRW', '€': 'EUR', '£': 'GBP',
                    '₽': 'RUB', '¥': 'JPY', '₹': 'INR'
                }
                
                # Check for Korean Won patterns
                if '원' in amount_text:
                    return 'KRW', amount_text
                
                for symbol, code in currency_symbols.items():
                    if symbol in amount_text:
                        return code, amount_text
                
                # Check for currency codes
                for code in ['USD', 'EUR', 'GBP', 'KRW', 'JPY', 'INR', 'RUB']:
                    if code in amount_text.upper():
                        return code, amount_text
                
                return None, amount_text
        
        return None, None

    def normalize_korean_text(self, text: str) -> str:
        """Normalize Korean text by removing spaces between Korean characters."""
        # Remove spaces between Korean characters (Hangul)
        # This handles cases like "보 냈 어" -> "보냈어"
        import re
        
        # Handle specific common patterns that OCR might separate
        common_patterns = {
            "보 냈 어": "보냈어",
            "보 냈 어요": "보냈어요",
            "보 냈 습니다": "보냈습니다",
            "송 금": "송금",
            "이 체": "이체",
            "결 제": "결제",
            "완 료": "완료",
            "성 공": "성공",
            "승 인": "승인",
            "확 인": "확인",
            "원 을": "원을",
            "만 원": "만원",
            "천 원": "천원",
            "억 원": "억원"
        }
        
        normalized_text = text
        
        # Apply common pattern replacements first
        for pattern, replacement in common_patterns.items():
            normalized_text = normalized_text.replace(pattern, replacement)
        
        # Pattern to match Korean characters with spaces between them
        # This will match sequences like "보 냈 어" or "원 을" etc.
        korean_with_spaces = re.findall(r'[가-힣]\s+[가-힣]+', normalized_text)
        
        for match in korean_with_spaces:
            # Remove spaces between Korean characters
            cleaned = re.sub(r'\s+', '', match)
            # Replace the original with cleaned version
            normalized_text = normalized_text.replace(match, cleaned)
        
        # Also handle cases where there might be multiple spaces between Korean chars
        # Pattern: Korean char + multiple spaces + Korean char
        korean_multi_spaces = re.findall(r'[가-힣]\s{2,}[가-힣]+', normalized_text)
        
        for match in korean_multi_spaces:
            # Remove all spaces between Korean characters
            cleaned = re.sub(r'\s+', '', match)
            # Replace the original with cleaned version
            normalized_text = normalized_text.replace(match, cleaned)
        
        # Final pass: remove any remaining spaces between Korean characters
        # This catches any patterns we might have missed
        normalized_text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', normalized_text)
        
        return normalized_text

    def analyze_payment_screenshot(self, image_path: str) -> PaymentInfo:
        """Analyze screenshot to determine if it's a payment/transfer screenshot."""
        
        print(f"Analyzing image: {image_path}")
        
        # Extract text using OCR
        ocr_results = self.extract_text_with_ocr(image_path)
        
        if not ocr_results:
            return PaymentInfo(
                status=PaymentStatus.UNKNOWN,
                confidence_score=0.0,
                detected_phrases=["No text detected"]
            )
        
        # Combine all extracted text
        full_text = " ".join([result.text for result in ocr_results])
        print(f"Extracted text: {full_text[:200]}...")  # Print first 200 chars
        
        # Initialize payment info
        payment_info = PaymentInfo(
            status=PaymentStatus.NOT_PAYMENT_SCREENSHOT,
            detected_phrases=[]
        )
        
        # Advanced payment detection using scoring system
        payment_score = self.calculate_payment_score(full_text)
        
        # Check for success patterns (English)
        success_matches = self.extract_patterns(full_text, self.success_patterns)
        if success_matches:
            payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
            payment_info.detected_phrases.extend(success_matches)
            payment_info.confidence_score = 0.9
        
        # Check for Korean success patterns
        korean_success_matches = self.extract_korean_patterns(full_text, self.korean_success_patterns)
        if korean_success_matches:
            payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
            payment_info.detected_phrases.extend(korean_success_matches)
            payment_info.confidence_score = 0.9
        
        # Check for failure patterns (English)
        failure_matches = self.extract_patterns(full_text, self.failure_patterns)
        if failure_matches:
            payment_info.status = PaymentStatus.PAYMENT_FAILED
            payment_info.detected_phrases.extend(failure_matches)
            payment_info.confidence_score = 0.9
        
        # Check for Korean failure patterns
        korean_failure_matches = self.extract_korean_patterns(full_text, self.korean_failure_patterns)
        if korean_failure_matches:
            payment_info.status = PaymentStatus.PAYMENT_FAILED
            payment_info.detected_phrases.extend(korean_failure_matches)
            payment_info.confidence_score = 0.9
        
        # Extract currency and amount
        currency, amount = self.detect_currency_and_amount(full_text)
        payment_info.currency = currency
        payment_info.amount = amount
        
        # Extract payment method (English)
        payment_method_matches = self.extract_patterns(full_text, self.payment_method_patterns)
        if payment_method_matches:
            payment_info.payment_method = payment_method_matches[0]
        
        # Extract Korean payment method
        korean_payment_method_matches = self.extract_korean_patterns(full_text, self.korean_payment_method_patterns)
        if korean_payment_method_matches:
            payment_info.payment_method = korean_payment_method_matches[0]
        
        # Extract transaction ID (English)
        for pattern in self.transaction_id_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                payment_info.transaction_id = match.group(1)
                break
        
        # Extract Korean transaction ID
        for pattern in self.korean_transaction_id_patterns:
            match = re.search(pattern, full_text)
            if match:
                payment_info.transaction_id = match.group(1)
                break
        
        # Check for Korean bank keywords to identify payment screenshots
        korean_bank_matches = self.extract_korean_patterns(full_text, self.korean_bank_keywords)
        if korean_bank_matches:
            payment_info.detected_phrases.extend(korean_bank_matches)
            if payment_info.status == PaymentStatus.NOT_PAYMENT_SCREENSHOT:
                payment_info.status = PaymentStatus.UNKNOWN
                payment_info.confidence_score = 0.7
        
        # Use advanced scoring for better detection
        if payment_score >= 3:
            if payment_info.status == PaymentStatus.NOT_PAYMENT_SCREENSHOT:
                payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
                payment_info.confidence_score = 0.85
                payment_info.detected_phrases.append(f'High payment score: {payment_score}')
            elif payment_info.status == PaymentStatus.UNKNOWN:
                payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
                payment_info.confidence_score = 0.85
                payment_info.detected_phrases.append(f'High payment score: {payment_score}')
        elif payment_score >= 2:
            if payment_info.status == PaymentStatus.NOT_PAYMENT_SCREENSHOT:
                payment_info.status = PaymentStatus.UNKNOWN
                payment_info.confidence_score = 0.7
                payment_info.detected_phrases.append(f'Medium payment score: {payment_score}')
            elif payment_info.status == PaymentStatus.UNKNOWN:
                payment_info.confidence_score = 0.7
                payment_info.detected_phrases.append(f'Medium payment score: {payment_score}')
        
        # Additional fallback: Check for partial Korean characters and payment-like patterns
        if payment_info.status == PaymentStatus.NOT_PAYMENT_SCREENSHOT:
            # Check for Korean characters (even partial ones)
            korean_chars = re.findall(r'[가-힣]', full_text)
            if korean_chars:
                # If we see Korean characters, it's likely a Korean app screenshot
                payment_info.status = PaymentStatus.UNKNOWN
                payment_info.confidence_score = 0.6
                payment_info.detected_phrases.append('Korean characters detected')
            
            # Check for large numbers that might be amounts (4+ digits)
            large_numbers = re.findall(r'\b\d{4,}\b', full_text)
            if large_numbers and len(korean_chars) > 0:
                # Korean text + large numbers = likely payment screenshot
                payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
                payment_info.confidence_score = 0.7
                payment_info.detected_phrases.append('Korean text + large numbers')
            
            # Check for common payment-related number patterns
            payment_numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*\b', full_text)
            if len(payment_numbers) >= 2 and len(korean_chars) > 0:
                # Multiple numbers + Korean text = likely payment
                payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
                payment_info.confidence_score = 0.7
                payment_info.detected_phrases.append('Multiple numbers + Korean text')
        
        # After extracting currency and amount, payment method, etc.
        # (Insert after extracting korean_bank_matches)
        # Check for fallback: bank name + KRW amount
        if payment_info.status == PaymentStatus.NOT_PAYMENT_SCREENSHOT:
            if payment_info.currency == 'KRW' and korean_bank_matches:
                payment_info.status = PaymentStatus.PAYMENT_SUCCESSFUL
                payment_info.confidence_score = 0.7
                payment_info.detected_phrases.append('KRW+BankName fallback')
        
        # If we found currency or payment indicators, it's likely a payment screenshot
        if currency or payment_method_matches or korean_payment_method_matches or any(keyword in full_text.lower() for keyword in ['payment', 'transfer', 'transaction', 'send', 'receive']):
            if payment_info.status == PaymentStatus.NOT_PAYMENT_SCREENSHOT:
                payment_info.status = PaymentStatus.UNKNOWN
                payment_info.confidence_score = 0.6
        
        return payment_info

    def calculate_payment_score(self, text: str) -> int:
        """Calculate a payment score based on multiple indicators (similar to ChatGPT's approach)."""
        score = 0
        
        # Normalize the text first to handle OCR artifacts
        normalized_text = self.normalize_korean_text(text)
        
        # Toss Bank specific keywords that strongly suggest payment confirmation
        toss_payment_keywords = [
            "보냈어요",  # "sent" in Korean - very common in Toss
            "보냈습니다",  # "sent" in Korean (formal)
            "송금",     # "transfer"
            "확인",     # "confirm"
            "원",       # Korean Won currency
            "Payment",
            "Sent",
            "to",
            "Success",
            "Amount",
            "Transaction",
            "confirmed",
            "Toss",
            "완료",     # Complete
            "성공",     # Success
            "승인",     # Approval
            "공유하기",  # Share button
            "메모 남기기",  # Leave a memo
            "확인",     # Confirm button
            "취소",     # Cancel button
            "보내기",   # Send
            "받기",     # Receive
            "결제",     # Payment
            "이체",     # Transfer
            "송금완료",  # Transfer completed
            "이체완료",  # Transfer completed
            "결제완료",  # Payment completed
            "처리완료",  # Processing completed
            "승인완료",  # Approval completed
            "확인완료"  # Confirmation completed
        ]
        
        # Partial Korean phrases that might be separated by OCR
        partial_korean_phrases = [
            "보 냈",  # "sent" (separated)
            "보냈",   # "sent" (partial)
            "송금",   # "transfer"
            "이체",   # "transfer"
            "결제",   # "payment"
            "완료",   # "complete"
            "성공",   # "success"
            "승인",   # "approval"
            "확인",   # "confirm"
            "원",     # "won" currency
            "만원",   # "10,000 won"
            "천원",   # "1,000 won"
            "억원"    # "100 million won"
        ]
        
        # Check for Korean Won amounts
        won_patterns = [
            r'[\d,]+원',
            r'[\d,]+\.?\d*원',
            r'[\d,]+,\d{3}원',
            r'[\d,]+\.?\d*만원',
            r'[\d,]+\.?\d*천원',
            r'[\d,]+\.?\d*억원',
            r'KRW\s*[\d,]+',
            r'₩[\d,]+'
        ]
        
        # Check for large numbers (likely amounts)
        large_numbers = re.findall(r'\b\d{4,}\b', text)
        
        # Count keyword matches (exact matches) - use normalized text
        for keyword in toss_payment_keywords:
            if keyword.lower() in normalized_text.lower():
                score += 1
        
        # Count partial Korean phrase matches (for OCR artifacts) - use original text
        for phrase in partial_korean_phrases:
            if phrase in text:
                score += 1
        
        # Count Won amount patterns
        for pattern in won_patterns:
            if re.search(pattern, text):
                score += 2  # Higher weight for currency amounts
                break
        
        # Count large numbers
        if len(large_numbers) >= 1:
            score += 1
        
        # Check for Korean characters (indicates Korean app)
        korean_chars = re.findall(r'[가-힣]', text)
        if len(korean_chars) >= 3:  # At least 3 Korean characters
            score += 1
        
        # Special bonus for "보냈" pattern (very common in Toss) - check both original and normalized
        if "보냈" in text or "보냈" in normalized_text:
            score += 2  # High weight for this specific pattern
        
        # Special bonus for "원" + large number pattern
        if "원" in text and len(large_numbers) >= 1:
            score += 1
        
        # Special bonus for normalized Korean phrases
        if "보냈어요" in normalized_text or "보냈습니다" in normalized_text:
            score += 3  # Very high weight for complete phrases
        
        return score

    def should_delete_screenshot(self, payment_info: PaymentInfo) -> bool:
        """Determine if screenshot should be marked for deletion based on analysis."""
        
        # Delete if it's a successful payment/transfer with high confidence
        if (payment_info.status == PaymentStatus.PAYMENT_SUCCESSFUL and 
            payment_info.confidence_score > 0.8):
            return True
        
        # Delete if currency is detected and has success indicators
        if (payment_info.currency and 
            payment_info.confidence_score > 0.7):
            return True
        
        return False

def main():
    """Main function to run the payment screenshot detector."""
    
    parser = argparse.ArgumentParser(description='Detect payment/transfer screenshots using OCR')
    parser.add_argument('image_path', help='Path to the screenshot image')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return
    
    # Initialize detector
    detector = PaymentScreenshotDetector()
    
    # Analyze the screenshot
    result = detector.analyze_payment_screenshot(args.image_path)
    
    # Display results
    print("\n" + "="*50)
    print("PAYMENT SCREENSHOT ANALYSIS RESULTS")
    print("="*50)
    
    print(f"Image: {args.image_path}")
    print(f"Status: {result.status.value}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    
    # Debug: Show payment score if verbose
    if args.verbose:
        detector = PaymentScreenshotDetector()
        ocr_results = detector.extract_text_with_ocr(args.image_path)
        if ocr_results:
            full_text = " ".join([result.text for result in ocr_results])
            payment_score = detector.calculate_payment_score(full_text)
            normalized_text = detector.normalize_korean_text(full_text)
            print(f"Payment Score: {payment_score}")
            print(f"Original text: {full_text}")
            print(f"Normalized text: {normalized_text}")
    
    if result.currency:
        print(f"Currency: {result.currency}")
    
    if result.amount:
        print(f"Amount: {result.amount}")
    
    if result.payment_method:
        print(f"Payment Method: {result.payment_method}")
    
    if result.transaction_id:
        print(f"Transaction ID: {result.transaction_id}")
    
    if result.detected_phrases:
        print(f"Detected Phrases: {', '.join(result.detected_phrases)}")
    
    # Deletion recommendation
    should_delete = detector.should_delete_screenshot(result)
    print(f"\nRecommend Deletion: {'YES' if should_delete else 'NO'}")
    
    if should_delete:
        print("Reason: This appears to be a completed payment/transfer screenshot")
    else:
        print("Reason: Not a completed payment screenshot or low confidence")
    
    # Save results to JSON if specified
    if args.output:
        result_dict = {
            'image_path': args.image_path,
            'status': result.status.value,
            'confidence_score': result.confidence_score,
            'currency': result.currency,
            'amount': result.amount,
            'payment_method': result.payment_method,
            'transaction_id': result.transaction_id,
            'detected_phrases': result.detected_phrases,
            'recommend_deletion': should_delete
        }
        
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()