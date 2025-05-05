import re
import string
from typing import List, Tuple, Dict, Any, Union


class PasswordValidator:
    """
    Password validation utility to ensure secure passwords.
    
    This class provides methods to validate password strength based on multiple criteria.
    """
    
    @staticmethod
    def validate(password: str, min_length: int = 8, require_uppercase: bool = True,
                 require_lowercase: bool = True, require_digit: bool = True,
                 require_special: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate password strength based on specified criteria.
        
        Args:
            password (str): The password to validate
            min_length (int, optional): Minimum required length. Defaults to 8.
            require_uppercase (bool, optional): Require at least one uppercase letter. Defaults to True.
            require_lowercase (bool, optional): Require at least one lowercase letter. Defaults to True.
            require_digit (bool, optional): Require at least one digit. Defaults to True.
            require_special (bool, optional): Require at least one special character. Defaults to True.
            
        Returns:
            Tuple[bool, List[str]]: A tuple containing validation result (bool) and list of error messages
        """
        errors = []
        
        # Check minimum length
        if len(password) < min_length:
            errors.append(f"Password must be at least {min_length} characters long.")
        
        # Check for uppercase letters
        if require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must include at least one uppercase letter.")
        
        # Check for lowercase letters
        if require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must include at least one lowercase letter.")
        
        # Check for digits
        if require_digit and not any(c.isdigit() for c in password):
            errors.append("Password must include at least one number.")
        
        # Check for special characters
        special_chars = set(string.punctuation)
        if require_special and not any(c in special_chars for c in password):
            errors.append("Password must include at least one special character (e.g., !@#$%^&*).")
        
        # Check if valid
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    @staticmethod
    def calculate_strength(password: str) -> Dict[str, Any]:
        """
        Calculate password strength score and provide feedback.
        
        Args:
            password (str): The password to evaluate
            
        Returns:
            Dict[str, Any]: Dictionary containing strength score (0-100) and feedback
        """
        if not password:
            return {"score": 0, "strength": "Very Weak", "feedback": ["Password is empty."]}
        
        score = 0
        feedback = []
        
        # Length contribution (up to 25 points)
        length_score = min(len(password) * 2, 25)
        score += length_score
        
        # Character diversity contribution (up to 50 points)
        has_uppercase = any(c.isupper() for c in password)
        has_lowercase = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in string.punctuation for c in password)
        
        diversity_score = 0
        if has_uppercase:
            diversity_score += 12.5
        else:
            feedback.append("Add uppercase letters to strengthen your password.")
            
        if has_lowercase:
            diversity_score += 12.5
        else:
            feedback.append("Add lowercase letters to strengthen your password.")
            
        if has_digit:
            diversity_score += 12.5
        else:
            feedback.append("Add numbers to strengthen your password.")
            
        if has_special:
            diversity_score += 12.5
        else:
            feedback.append("Add special characters to strengthen your password.")
            
        score += diversity_score
        
        # Complexity contribution (up to 25 points)
        # Check for sequential patterns
        has_sequential = (
            re.search(r'abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz', 
                     password.lower()) or
            re.search(r'012|123|234|345|456|567|678|789', password)
        )
        
        # Check for repeated characters
        has_repeated = re.search(r'(.)\1\1', password)  # Three or more of the same character
        
        # Check for keyboard patterns
        keyboard_patterns = ['qwerty', 'asdfgh', 'zxcvbn', '123456', 'qweasd']
        has_keyboard_pattern = any(pattern in password.lower() for pattern in keyboard_patterns)
        
        complexity_score = 25
        if has_sequential:
            complexity_score -= 10
            feedback.append("Avoid sequential characters (e.g., abc, 123).")
            
        if has_repeated:
            complexity_score -= 10
            feedback.append("Avoid repeating the same character multiple times.")
            
        if has_keyboard_pattern:
            complexity_score -= 10
            feedback.append("Avoid common keyboard patterns (e.g., qwerty).")
            
        score += max(complexity_score, 0)
        
        # Determine strength category
        strength = "Very Weak"
        if score >= 80:
            strength = "Very Strong"
        elif score >= 60:
            strength = "Strong"
        elif score >= 40:
            strength = "Moderate"
        elif score >= 20:
            strength = "Weak"
            
        # Add general feedback if password is weak
        if score < 40:
            feedback.append("Consider using a longer password with a mix of character types.")
        
        return {
            "score": score,
            "strength": strength,
            "feedback": feedback
        }
    
    @staticmethod
    def is_common_password(password: str, common_passwords: Union[List[str], None] = None) -> bool:
        """
        Check if the password is among commonly used (and thus insecure) passwords.
        
        Args:
            password (str): The password to check
            common_passwords (List[str], optional): List of common passwords to check against. 
                                                   If None, uses a small built-in list.
                                                   
        Returns:
            bool: True if the password is common, False otherwise
        """
        if common_passwords is None:
            # Small sample of common passwords
            common_passwords = [
                "password", "123456", "12345678", "qwerty", "admin", 
                "welcome", "123456789", "1234567", "password1", "abc123",
                "111111", "123123", "admin123", "qwerty123", "1q2w3e4r",
                "654321", "555555", "lovely", "7777777", "welcome1",
                "888888", "princess", "dragon", "password123", "sunshine",
                "master", "hottie", "flower", "loveme", "zaq1zaq1"
            ]
        
        return password.lower() in [p.lower() for p in common_passwords]