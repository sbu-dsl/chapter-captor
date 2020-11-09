import inflect
from collections import Counter
from collections import OrderedDict
import collections
import re
import pandas as pd

def get_regex_rule(number_type):
    
    p = inflect.engine()
    
    numeral = "[0-9]+"
    
    roman_upper = "(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})(?![A-Za-z0-9'\"])"
    roman_lower = "(?=[mdxlxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})(?![A-Za-z0-9'\"])"
    
    word_number_one = '|'.join(sorted([p.number_to_words(x) for x in range(201)], key=len)[::-1])
    word_number_One = '|'.join(sorted([p.number_to_words(x).title() for x in range(201)], key=len)[::-1])
    word_number_ONE = '|'.join(sorted([p.number_to_words(x).upper() for x in range(201)], key=len)[::-1])
    word_number_first = '|'.join(sorted([p.ordinal(p.number_to_words(x)) for x in range(201)], key=len)[::-1])
    word_number_First = '|'.join(sorted([p.ordinal(p.number_to_words(x)).title() for x in range(201)], key=len)[::-1])
    word_number_FIRST = '|'.join(sorted([p.ordinal(p.number_to_words(x)).upper() for x in range(201)], key=len)[::-1])

    tmp = [p.number_to_words(x) for x in range(201)]
    tmp2 = list()
    for elem in tmp:
        if '-' in elem:
            tmp2.append(elem.replace('-', '[\s]+-[\s]+'))
    word_number_one += '|' + '|'.join(tmp2)

    tmp = [p.number_to_words(x).title() for x in range(201)]
    tmp2 = list()
    for elem in tmp:
        if '-' in elem:
            tmp2.append(elem.replace('-', '[\s]+-[\s]+'))
    word_number_One += '|' + '|'.join(tmp2)

    tmp = [p.number_to_words(x).upper() for x in range(201)]
    tmp2 = list()
    for elem in tmp:
        if '-' in elem:
            tmp2.append(elem.replace('-', '[\s]+-[\s]+'))
    word_number_ONE += '|' + '|'.join(tmp2)

    tmp = [p.ordinal(p.number_to_words(x)) for x in range(201)]
    tmp2 = list()
    for elem in tmp:
        if '-' in elem:
            tmp2.append(elem.replace('-', '[\s]+-[\s]+'))
    word_number_first += '|' + '|'.join(tmp2)

    tmp = [p.ordinal(p.number_to_words(x)).title() for x in range(201)]
    tmp2 = list()
    for elem in tmp:
        if '-' in elem:
            tmp2.append(elem.replace('-', '[\s]+-[\s]+'))
    word_number_First += '|' + '|'.join(tmp2)

    tmp = [p.ordinal(p.number_to_words(x)).upper() for x in range(201)]
    tmp2 = list()
    for elem in tmp:
        if '-' in elem:
            tmp2.append(elem.replace('-', '[\s]+-[\s]+'))
    word_number_FIRST += '|' + '|'.join(tmp2)
    
    string_to_pattern = dict()
    string_to_pattern['numeral'] = numeral
    string_to_pattern['roman_upper'] = roman_upper
    string_to_pattern['roman_lower'] = roman_lower
    string_to_pattern['word_number_one'] = word_number_one
    string_to_pattern['word_number_One'] = word_number_One
    string_to_pattern['word_number_ONE'] = word_number_ONE
    string_to_pattern['word_number_first'] = word_number_first
    string_to_pattern['word_number_First'] = word_number_First
    string_to_pattern['word_number_FIRST'] = word_number_FIRST
    
    return re.compile(string_to_pattern[number_type])


def convert_to_int(number, number_type):
    
    def convert_one_to_int(number):
        p = inflect.engine()
        #number = number.replace(' ', '')
        number = ' '.join(number.strip().split())
        number = '-'.join([x.strip() for x in number.split('-')])
        l = [p.number_to_words(x) for x in range(201)]
        return l.index(number)
    
    def convert_first_to_int(number):
        p = inflect.engine()
        #number = number.replace(' ', '')
        number = ' '.join(number.strip().split())
        number = '-'.join([x.strip() for x in number.split('-')])
        l = [p.ordinal(p.number_to_words(x)) for x in range(201)]
        return l.index(number)
    
    def convert_roman_to_int(number):
        
        def value(r): 
            if (r == 'I'): 
                return 1
            if (r == 'V'): 
                return 5
            if (r == 'X'): 
                return 10
            if (r == 'L'): 
                return 50
            if (r == 'C'): 
                return 100
            if (r == 'D'): 
                return 500
            if (r == 'M'): 
                return 1000
            return -1
        res = 0
        i = 0
        
        number = number.upper()
        while (i < len(number)): 
            # Getting value of symbol s[i]
            s1 = value(number[i])
            if (i+1 < len(number)):
                # Getting value of symbol s[i+1]
                s2 = value(number[i+1])
                # Comparing both values
                if (s1 >= s2):
                    # Value of current symbol is greater
                    # or equal to the next symbol
                    res = res + s1
                    i = i + 1
                else:
                    # Value of current symbol is greater
                    # or equal to the next symbol
                    res = res + s2 - s1
                    i = i + 2
            else:
                res = res + s1
                i = i + 1
        return res
    
    if number_type == 'numeral':
        return int(number)
    if number_type.lower() == 'word_number_one':
        return convert_one_to_int(number.lower())
    if number_type.lower() == 'word_number_first':
        return convert_first_to_int(number.lower())
    if number_type.startswith('roman'):
        return convert_roman_to_int(number)
    
    return 0



def findLISindices(arrA):
    LIS = [0 for i in range(len(arrA))]
    for i in range(len(arrA)):
        maximum = -1
        for j in range(i):
            if arrA[i] > arrA[j]:
                if maximum == -1 or maximum < LIS[j] + 1:
                    maximum = 1 + LIS[j]
        if maximum == -1:
            maximum = 1
        LIS[i] = maximum
    
    result = -1
    index = -1
    
    for i in range(len(LIS)):
        if result < LIS[i]:
            result = LIS[i]
            index = i
            
    answer = list()
    answer.insert(0, index)
    res = result - 1
    for i in range(index - 1, -1, -1):
        if LIS[i] == res:
            answer.insert(0, i)
            res -= 1
    
    return answer


def write_roman(num):

    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])

def convert_to_number_type_regex_string(x, number_type):
    if number_type == 'numeral':
        return str(x) + "(?![A-Za-z0-9'\"])"
    
    if number_type == 'roman_upper':
        return write_roman(x) + "(?![A-Za-z0-9'\"])"
    elif number_type == 'roman_lower':
        return write_roman(x).lower() + "(?![A-Za-z0-9'\"])"
    
    p = inflect.engine()
    conv = p.number_to_words(x)

    if number_type == 'word_number_One':
        conv = conv.title()
    elif number_type == 'word_number_ONE':
        conv = conv.upper()

    elif number_type == 'word_number_first':
        conv = p.ordinal(conv)
    elif number_type == 'word_number_First':
        conv = p.ordinal(conv).title()
    elif number_type == 'word_number_FIRST':
        conv = p.ordinal(conv).upper()

    if '-' in conv:
        conv = conv.replace('-', '[\s]*-[\s]*')

    return conv + "(?![A-Za-z0-9'\"])"


def find_in_book(tokens, search_pattern, from_index, to_index, last_occurrence=False):
    curr = 0
    char_indices = list()
    for elem in tokens:
        char_indices.append(curr)
        if elem == '[unused1]':
            curr += 2
        else:
            curr += 1 + len(elem)
    
    text = ' '.join([x if x != '[unused1]' else '\n' for x in tokens])
    
    if from_index >= len(char_indices):
        return []
    if to_index >= len(char_indices):
        to_index = len(char_indices) - 1
    
    lower = char_indices[from_index]
    upper = min(char_indices[to_index], len(text)) + 1
    ans = list()
    for elem in search_pattern.finditer(text[lower:upper]):
        if elem.span()[0] + char_indices[from_index] in char_indices and text[elem.span()[0] + char_indices[from_index] - 2] == '\n':
            ans.append(elem)

    ans2 = list()
    
    for m in ans:
            
        start, end = m.span()
        start += char_indices[from_index]
        end += char_indices[from_index]
    
        if start in char_indices:
            start_idx = char_indices.index(start)
            idx = start_idx
            while idx < len(char_indices) and char_indices[idx] < end:
                idx += 1
            if idx - 2 != start_idx:
                ans2.append([x for x in range(start_idx, idx)])
            ans2.append([start_idx])

    return ans2

def findRestartIndices(nums):
    if len(nums) == 0:
        return []
    ans = [0]
    if len(nums) == 1:
        return ans
    
    idx = 1
    while idx < len(nums):
        if nums[idx] <= nums[idx - 1]:
            return ans
        ans.append(idx)
        idx += 1
    return ans

def hunt(pred_texts, pred_indices, matched_rule, tokens, body_start_token_idx, number_restart=False):
    # Get the number type from the matched rule
    number_type = matched_rule[-1]
    
    # Get the regex rule corresponding to the number type
    number_regex_rule = get_regex_rule(number_type)
    
    # Get the position of the number type in the matched rule
    number_index = matched_rule.index(number_type)
    
    number_match_dict = collections.defaultdict(lambda: list())
    found_missing_dict = collections.defaultdict(lambda: list())
    indices_groups_dict = collections.defaultdict(lambda: list())
    
    if number_index > 0:
        # Rule for descriptor that may occur before number
        desc = "(CHAPTER|Chapter|CHAP|Chap|PART|Part|BOOK|Book|STORY|Story|LETTER|Letter|VOLUME|Volume|VOL|Vol|CASE|Case|THE|The)[\s]*(THE|The|the|NO|No|no|NO\.|No\.|no\.|NUMBER|Number|number|NUMBER\.|Number\.|number\.)*[\s]*"
        desc_rule = re.compile(desc)
        for idx, text in enumerate(pred_texts):
            desc_match = desc_rule.match(text)
            if desc_match:
                number_start = desc_match.span()[1]
                matched_text = text[:number_start]
                rem_header = text[number_start:]
                number_match = number_regex_rule.match(rem_header)
                if number_match:
                    start, end = number_match.span()
                    number_match_dict[matched_text].append(rem_header[start:end])
                    indices_groups_dict[matched_text].append(pred_indices[idx])
    
    else:
        for idx, text in enumerate(pred_texts):
            number_match = number_regex_rule.match(text)
            if number_match:
                start, end = number_match.span()
                number_match_dict[''].append(text[start:end])
                indices_groups_dict[''].append(pred_indices[idx])
    
    seen = set()
    for group in pred_indices:
        seen.update(group)
    for descriptor in number_match_dict:
        pred_indices_tmp = indices_groups_dict[descriptor]
        numbers = number_match_dict[descriptor]
        converted_numbers = [convert_to_int(n, number_type) for n in numbers]
        
        queue = list()
        queue.append((converted_numbers, 0, body_start_token_idx, len(tokens) - 1))
        
        while queue:
            converted_numbers, offset, last_from, last_to = queue.pop(0)
            if number_restart:
                lis_indices = findRestartIndices(converted_numbers)
            else:
                lis_indices = findLISindices(converted_numbers)
            
            if len(lis_indices) > 0:
                if lis_indices[0] > 0:
                    a = converted_numbers[:lis_indices[0]]
                    b = offset
                    c = last_from
                    d = pred_indices_tmp[offset + lis_indices[0]][-1]
                    queue.append((a, b, c, d))
                if lis_indices[-1] < len(converted_numbers) - 1:
                    a = converted_numbers[lis_indices[-1] + 1:]
                    b = offset + lis_indices[-1] + 1
                    c = pred_indices_tmp[offset + lis_indices[-1]][-1] + 1
                    d = last_to
                    queue.append((a, b, c, d))
            
            smallest_number = converted_numbers[lis_indices[0]] - 1
            while smallest_number > 0:
                from_index = last_from
                to_index = pred_indices_tmp[offset + lis_indices[0]][-1]
                if number_index > 0:
                    search_pattern = descriptor.strip() + '[\s]*' + convert_to_number_type_regex_string(smallest_number, number_type)
                else:
                    search_pattern = convert_to_number_type_regex_string(smallest_number, number_type)
                search_pattern = re.compile(search_pattern)
                found_indices = find_in_book(tokens, search_pattern, from_index, to_index)
                if len(found_indices) > 0:
                    for group in found_indices:
                        if not any(x in seen for x in group):
                            found_missing_dict[descriptor].insert(0, group)
                            seen.update(group)
                    to_index = found_indices[-1][0] - 1
                else:
                    break
                smallest_number -= 1
                
            idx = 0
            for idx in range(len(lis_indices) - 1):
                temp = converted_numbers[lis_indices[idx]] + 1
                from_index = pred_indices_tmp[offset + lis_indices[idx]][-1] + 1
                to_index = pred_indices_tmp[offset + lis_indices[idx + 1]][-1]
                while converted_numbers[lis_indices[idx + 1]] > temp:
                    if number_index > 0:
                        search_pattern = descriptor.strip() + '[\s]*' + convert_to_number_type_regex_string(temp, number_type)
                    else:
                        search_pattern = convert_to_number_type_regex_string(temp, number_type)
                    search_pattern = re.compile(search_pattern)
                    found_indices = find_in_book(tokens, search_pattern, from_index, to_index)
                    if len(found_indices) > 0:
                        for group in found_indices:
                            if not any(x in seen for x in group):
                                found_missing_dict[descriptor].append(group)
                                seen.update(group)
                        from_index = found_indices[0][-1] + 1
                    temp += 1
             
            largest_number = converted_numbers[lis_indices[-1]]
            while True:
                largest_number += 1
                from_index =  pred_indices_tmp[offset + lis_indices[-1]][-1] + 1
                to_index = last_to
                if number_index > 0:
                    search_pattern = descriptor.strip() + '[\s]*' + convert_to_number_type_regex_string(largest_number, number_type)
                else:
                    search_pattern = convert_to_number_type_regex_string(largest_number, number_type)
                search_pattern = re.compile(search_pattern)
                found_indices = find_in_book(tokens, search_pattern, from_index, to_index)
                if len(found_indices) > 0:
                    for group in found_indices:
                        if not any(x in seen for x in group):
                            found_missing_dict[descriptor].append(group)
                            seen.update(group)
                    from_index = found_indices[0][-1] + 1
                else:
                    break
    ans = list()
    for descriptor in found_missing_dict:
        for group in found_missing_dict[descriptor]:
            ans.append(group)
    return ans
                
def find_missing_chapters(df, pred_indices, matched_rule, body_start_token_idx):
    number_list = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']
    
    # If the matched rule does not contain a number form, return as is
    if len(set(number_list).intersection(set(matched_rule))) == 0:
        return pred_indices
    
    
    tokens = [str(x) for x in list(df['token'])]
    line_nums = list(df['line_number'])
    pred_texts = [' '.join([tokens[x] if tokens[x] != '[unused1]' else '\n' for x in group]) for group in pred_indices]
    
    missing_indices = hunt(pred_texts, pred_indices, matched_rule, tokens, body_start_token_idx, number_restart=False)
    
    new_pred_indices = pred_indices + missing_indices
    new_pred_texts = [' '.join([tokens[x] if tokens[x] != '[unused1]' else '\n' for x in group]) for group in new_pred_indices]
    
    new_missing_indices = hunt(new_pred_texts, new_pred_indices, matched_rule, tokens, body_start_token_idx, number_restart=True)
    
    return sorted(pred_indices + missing_indices + new_missing_indices)
