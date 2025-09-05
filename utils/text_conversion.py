from num2words import num2words

class TextConverter:
    """Utility class for converting amounts to words in different languages"""
    
    def amount_to_words_english(self, amount):
        """Convert amount to English words for USD"""
        dollars = int(amount)
        cents = round((amount - dollars) * 100)
        
        if dollars == 0:
            dollar_text = ""
        elif dollars == 1:
            dollar_text = "one dollar"
        else:
            dollar_text = f"{num2words(dollars)} dollars"
        
        if cents == 0:
            cent_text = ""
        elif cents == 1:
            cent_text = "one cent"
        else:
            cent_text = f"{num2words(cents)} cents"
        
        if dollar_text and cent_text:
            return f"{dollar_text} and {cent_text}"
        elif dollar_text:
            return dollar_text
        else:
            return cent_text if cent_text else "zero dollars"
    
    def amount_to_words_khmer(self, amount):
        """Convert amount to words for KHR (convert to integer riels)"""
        riels = int(amount)  # Convert to integer, no decimals for KHR
        
        if riels == 0:
            return "zero riels"
        elif riels == 1:
            return "one riel"
        else:
            return f"{num2words(riels)} riels"
    
    def number_to_chinese(self, num):
        """Comprehensive Chinese number conversion for numbers up to billions"""
        if num == 0:
            return "零"
        
        digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        
        def convert_section(n, is_beginning=True):
            """Convert a section (0-9999) to Chinese
            is_beginning: True if this is the first/leading section of the number
            """
            if n == 0:
                return ""
            elif n < 10:
                return digits[n]
            elif n < 100:
                tens = n // 10
                ones = n % 10
                if tens == 1:
                    # Special case: if this is not the beginning section, use "一十"
                    if is_beginning:
                        return "十" + (digits[ones] if ones > 0 else "")
                    else:
                        return "一十" + (digits[ones] if ones > 0 else "")
                else:
                    return digits[tens] + "十" + (digits[ones] if ones > 0 else "")
            elif n < 1000:
                hundreds = n // 100
                remainder = n % 100
                result = digits[hundreds] + "百"
                if remainder == 0:
                    return result
                elif remainder < 10:
                    return result + "零" + digits[remainder]
                else:
                    # When we have remainder in tens, it's not the beginning anymore
                    return result + convert_section(remainder, False)
            else:  # n < 10000
                thousands = n // 1000
                remainder = n % 1000
                result = digits[thousands] + "千"
                if remainder == 0:
                    return result
                elif remainder < 100:
                    if remainder < 10:
                        return result + "零" + convert_section(remainder, False)
                    else:
                        # Special handling for 10-99 after thousands
                        return result + "零" + convert_section(remainder, False)
                else:
                    return result + convert_section(remainder, False)
        
        # Handle different ranges
        if num < 10000:
            return convert_section(num, True)
        elif num < 100000000:  # Less than 1 yi (100 million)
            wan = num // 10000
            remainder = num % 10000
            result = convert_section(wan, True) + "万"
            if remainder == 0:
                return result
            elif remainder < 1000:
                if remainder < 100:
                    if remainder < 10:
                        return result + "零" + convert_section(remainder, False)
                    else:
                        # Special case: numbers like 50010, 30010 - need "零一十"
                        return result + "零" + convert_section(remainder, False)
                else:
                    return result + "零" + convert_section(remainder, False)
            else:
                return result + convert_section(remainder, False)
        else:  # 1 yi or more
            yi = num // 100000000
            remainder = num % 100000000
            result = convert_section(yi, True) + "亿"
            if remainder == 0:
                return result
            elif remainder < 10000000:  # Less than 1000 wan
                if remainder < 10000:
                    if remainder < 1000:
                        if remainder < 100:
                            if remainder < 10:
                                return result + "零" + convert_section(remainder, False)
                            else:
                                return result + "零" + convert_section(remainder, False)
                        else:
                            return result + "零" + convert_section(remainder, False)
                    else:
                        return result + "零" + convert_section(remainder, False)
                else:
                    wan_part = remainder // 10000
                    final_remainder = remainder % 10000
                    wan_result = convert_section(wan_part, False) + "万"
                    if final_remainder == 0:
                        return result + wan_result
                    elif final_remainder < 1000:
                        if final_remainder < 100:
                            if final_remainder < 10:
                                return result + wan_result + "零" + convert_section(final_remainder, False)
                            else:
                                return result + wan_result + "零" + convert_section(final_remainder, False)
                        else:
                            return result + wan_result + "零" + convert_section(final_remainder, False)
                    else:
                        return result + wan_result + convert_section(final_remainder, False)
            else:
                return result + self.number_to_chinese(remainder)
    
    def amount_to_words_chinese(self, amount, currency):
        """Convert amount to Chinese words"""
        try:
            if currency == "USD":
                # Handle decimal point format like 10.10 -> 十点一美元
                amount_str = f"{amount:.2f}"
                if '.' in amount_str:
                    whole_part, decimal_part = amount_str.split('.')
                    whole_num = int(whole_part)
                    
                    # Convert decimal part digit by digit (including leading zeros)
                    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
                    decimal_digits = []
                    
                    # Skip if all decimal digits are zero (like 1000.00)
                    if decimal_part == "00":
                        decimal_digits = []
                    else:
                        for digit in decimal_part:
                            decimal_digits.append(digits[int(digit)] + ",")
                    
                    if whole_num == 0 and not decimal_digits:
                        return "零美元"
                    elif whole_num == 0:
                        return f"零点{''.join(decimal_digits)}美元"
                    elif not decimal_digits:
                        return f"{self.number_to_chinese(whole_num)}美元"
                    else:
                        return f"{self.number_to_chinese(whole_num)}点{''.join(decimal_digits)}美元"
                else:
                    return f"{self.number_to_chinese(int(amount))}美元"
            
            else:  # KHR
                riels = int(amount)
                if riels == 0:
                    return "零[](/ʐweɪ˥˩)(/ɑɚ˨˩˦)"
                else:
                    return f"{self.number_to_chinese(riels)}[](/ʐweɪ˥˩)(/ɑɚ˨˩˦)"
        except Exception as e:
            print(f"Error in Chinese conversion: {e}")
            # Fallback to simple format
            if currency == "USD":
                return f"{amount}美元"
            else:
                return f"{int(amount)}[](/ʐweɪ˥˩)(/ɑɚ˨˩˦)"