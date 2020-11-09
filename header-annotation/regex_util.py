import inflect
from collections import Counter
import re
from itertools import groupby

def generate_sequences(l):
    if len(l) == 0:
        return []
    subsequent = generate_sequences(l[1:])
    answer = list()
    if len(subsequent) > 0:
        answer += subsequent
    for elem in l[0]:
        answer.append([elem])
        for elem2 in subsequent:
            answer.append([elem] + elem2)
    return answer

def remove_duplicates(l):
    res = [] 
    for i in l: 
        if i not in res: 
            res.append(i)
    return res

def remove_consecutives(l):
    res = list()
    for elem in l:
        tmp = [x[0] for x in groupby(elem)]
        if tmp not in res:
            res.append(tmp)
    return res

def remove_whitespace_from_ends(l):
    res = list()
    for elem in l:
        start = 0
        while start < len(elem) and elem[start] == 'whitespace':
            start += 1
        end = len(elem) - 1
        while end >= 0 and elem[end] == 'whitespace':
            end -= 1
        tmp = elem[start:end + 1]
        if tmp and tmp not in res:
            res.append(elem[start:end + 1])
    return res

def issublist(b, a):
    return b in [a[i:len(b)+i] for i in range(len(a))]
        
def remove_incompatible_consecutives(l, incompatible):
    inc_list = list()
    for x in incompatible:
        for y in incompatible:
            inc_list.append([x, y])
    res = list()
    for elem in l:
        if any([issublist(x, elem) for x in inc_list]):
            pass
        else:
            res.append(elem)
    return res


def get_corresponding_rule_regex_text(rule_name):
    word_numbers = ['word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']
    
    if rule_name == 'desc':
        return "[A-Za-z0-9\S]*(CHAPTER|Chapter|CHAP|Chap|PART|Part|BOOK|Book|STORY|Story|LETTER|Letter|VOLUME|Volume|VOL|Vol|CASE|Case|THE|The)[\s]*(THE|The|the|NO|No|no|NO\.|No\.|no\.|NUMBER|Number|number|NUMBER\.|Number\.|number\.)*"
    
    elif rule_name == 'roman_upper':
        return "(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})(?![A-Za-z0-9'\"])"
    elif rule_name == 'roman_lower':
        return "(?=[mdxlxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})(?![A-Za-z0-9'\"])"
    
    elif rule_name == 'numeral':
        return "[0-9]+"
    
    elif rule_name == 'punctuation':
        return "(?=\S)[^a-zA-Z\d\s]+"
    
    elif rule_name == 'title_upper':
        return "([^a-z\s\.]*(?=.*[A-Z]+.*)[^a-z]+[\n]+)+"
    elif rule_name == 'title_lower':
        return "((?=.*[a-z]+.*)(?=.*[A-Z]+.*)[^\r\n]*)\n[^\S\n]*\n"
    
    elif rule_name == 'whitespace':
        return "[\s]+"
    
    elif rule_name in word_numbers:
        p = inflect.engine()
        if rule_name == 'word_number_one':
            tmp = [p.number_to_words(x) for x in range(201)]
        elif rule_name == 'word_number_One':
            tmp = [p.number_to_words(x).title() for x in range(201)]
        elif rule_name == 'word_number_ONE':
            tmp = [p.number_to_words(x).upper() for x in range(201)]
        elif rule_name == 'word_number_first':
            tmp = [p.ordinal(p.number_to_words(x)) for x in range(201)]
        elif rule_name == 'word_number_First':
            tmp = [p.ordinal(p.number_to_words(x)).title() for x in range(201)]
        elif rule_name == 'word_number_FIRST':
            tmp = [p.ordinal(p.number_to_words(x)).upper() for x in range(201)]
        
        tmp2 = list()
        for elem in tmp:
            if '-' in elem:
                tmp2.append(elem.replace('-', '[\s]*-[\s]*'))
            else:
                tmp2.append(elem)
        reg = '|'.join(sorted(tmp2, key=len)[::-1])
        return reg
    
    return None
    


def get_rules():
    p = inflect.engine()

    desc = get_corresponding_rule_regex_text('desc')

    roman_upper = get_corresponding_rule_regex_text('roman_upper')
    roman_lower = get_corresponding_rule_regex_text('roman_lower')

    numeral = get_corresponding_rule_regex_text('numeral')

    punctuation = get_corresponding_rule_regex_text('punctuation')

    title_upper = get_corresponding_rule_regex_text('title_upper')
    title_lower = get_corresponding_rule_regex_text('title_lower')

    whitespace = get_corresponding_rule_regex_text('whitespace')

    word_number_one = get_corresponding_rule_regex_text('word_number_one')
    word_number_One = get_corresponding_rule_regex_text('word_number_One')
    word_number_ONE = get_corresponding_rule_regex_text('word_number_ONE')
    word_number_first = get_corresponding_rule_regex_text('word_number_first')
    word_number_First = get_corresponding_rule_regex_text('word_number_First')
    word_number_FIRST = get_corresponding_rule_regex_text('word_number_FIRST')


    l2 = list()

    l2.append(['desc'])
    l2.append(['whitespace'])
    l2.append(['punctuation'])
    l2.append(['whitespace'])
    l2.append(['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST'])
    l2.append(['whitespace'])
    l2.append(['punctuation'])
    l2.append(['whitespace'])
    l2.append(['title_upper', 'title_lower'])


    string_to_pattern = dict()
    string_to_pattern['desc'] = desc
    string_to_pattern['roman_upper'] = roman_upper
    string_to_pattern['roman_lower'] = roman_lower
    string_to_pattern['numeral'] = numeral
    string_to_pattern['punctuation'] = punctuation
    string_to_pattern['title_upper'] = title_upper
    string_to_pattern['title_lower'] = title_lower
    string_to_pattern['whitespace'] = whitespace

    string_to_pattern['word_number_one'] = word_number_one
    string_to_pattern['word_number_One'] = word_number_One
    string_to_pattern['word_number_ONE'] = word_number_ONE
    string_to_pattern['word_number_first'] = word_number_first
    string_to_pattern['word_number_First'] = word_number_First
    string_to_pattern['word_number_FIRST'] = word_number_FIRST


    all_seqs = generate_sequences(l2)
    all_seqs_2 = remove_duplicates(all_seqs)
    all_seqs_3 = remove_consecutives(all_seqs_2)
    all_seqs_4 = remove_whitespace_from_ends(all_seqs_3)

    all_seqs_5 = remove_incompatible_consecutives(all_seqs_4, ['title_upper', 'title_lower', 'roman_upper', 'roman_lower'])
    blacklist_sequences = list()
    blacklist_sequences.append(['desc', 'title_upper'])
    blacklist_sequences.append(['punctuation', 'title_lower'])
    blacklist_sequences.append(['punctuation'])
    blacklist_sequences.append(['punctuation', 'whitespace', 'punctuation'])
    blacklist_sequences.append(['punctuation', 'whitespace', 'title_upper'])
    blacklist_sequences.append(['punctuation', 'whitespace', 'title_lower'])
    blacklist_sequences.append(['punctuation', 'whitespace', 'punctuation', 'whitespace', 'title_lower'])
    blacklist_sequences.append(['title_lower'])
    blacklist_sequences.append(['roman_lower', 'whitespace', 'title_lower'])

    all_seqs_5 = [x for x in all_seqs_5 if x not in blacklist_sequences]

    all_seqs_5 = [x for x in all_seqs_5 if not issublist(['punctuation', 'whitespace', 'punctuation'], x)]
    
    number_list = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']
    all_seqs_5 = [x for x in all_seqs_5 if 'desc' not in x or ('desc' in x and any([m in x for m in number_list]))]
    
    all_seqs_5 = [x for x in all_seqs_5 if x[0] != 'word_number_one' and x[0] != 'word_number_first']

    all_seqs_5 = [x for x in all_seqs_5 if x[0] != 'punctuation' or (x[0] == 'punctuation' and 'punctuation' in x[1:])]
    
    
    all_seqs_5 = [x for x in all_seqs_5 if len(x) < 2 or not(x[-1] == 'title_lower' and x[-2] == 'whitespace')]
    
    
    tmp_all_seqs = list()
    first_list = ['word_number_First', 'word_number_FIRST']
    for seq in all_seqs_5:
        if 'desc' in seq:
            if 'word_number_First' in seq:
                word = 'word_number_First'
            elif 'word_number_FIRST' in seq:
                word = 'word_number_FIRST'
            else:
                tmp_all_seqs.append(seq)
                continue
            tmp_seq = list()
            for r in seq:
                if r == 'desc':
                    tmp_seq.append(word)
                elif r in first_list:
                    tmp_seq.append('desc')
                else:
                    tmp_seq.append(r)
            tmp_all_seqs.append(tmp_seq)
        tmp_all_seqs.append(seq)
    all_seqs_5 = tmp_all_seqs
    
    words = ['desc', 'title_upper', 'title_lower', 'word_number_first', 'word_number_First', 'word_number_FIRST', 'word_number_one', 'word_number_One', 'word_number_ONE', 'roman_upper', 'roman_lower']

    seqs_new = list()
    for s in all_seqs_5:
        b = ''.join(['1' if x in words else '0' for x in s])
        if '11' not in b:
            seqs_new.append(s)
    all_seqs_5 = seqs_new
    
    priority = ['desc', 'roman_upper', 'roman_lower', 'numeral', 'word_number_first', 'word_number_First', 'word_number_FIRST', 'word_number_one', 'word_number_One', 'word_number_ONE', 'whitespace', 'title_upper', 'title_lower', 'punctuation', '']
    
    # Make all rule sequences of equal length by appending empty strings
    m = max([len(x) for x in all_seqs_5])
    all_seqs_5_new = list()
    for elem in all_seqs_5:
        all_seqs_5_new.append(elem + [''] * (m - len(elem)))
    all_seqs_5 = all_seqs_5_new
    # Sort the rules found using pre-defined priority
    all_seqs_5.sort(key=lambda x:[priority.index(y) for y in x])
    # Remove the empty strings we appended earlier
    all_seqs_no_empty = [[elem for elem in x if elem] for x in all_seqs_5]
    
    rule_texts = [''.join(['(' + string_to_pattern[x] + ')' for x in y]) for y in all_seqs_no_empty]
    
    rules = [re.compile(x) for x in rule_texts]
    
    
    return all_seqs_5, all_seqs_no_empty, rules, priority


def get_best_matching_rule(text, text_rules, regex_rules, priority):
    answers = list()
    for idx in range(len(regex_rules)):
        r = regex_rules[idx]
        m = r.match(text)
        if m:
            if m.span()[0] == 0 and len(text[m.span()[1]:].strip()) == 0:
                answers.append(text_rules[idx])
    answers.sort(key=lambda x:[priority.index(y) for y in x])
    
    return answers


def get_best_rules_for_all(texts, sequences_as_lists, rules, priority):
    rules_found = list()
    for idx, text in enumerate(texts):
        ans = get_best_matching_rule(text, sequences_as_lists, rules, priority)
        if len(ans) > 0:
            rules_found.append(ans[0])
        else:
            rules_found.append(None)
    if len(rules_found) == 0:
        return None
    return rules_found



def find_highest_priority_rule(rules_found):
    c = Counter(rules_found)
    for rule in rules_found:
        if c[rule] > 1:
            return rule
    return rules_found[0]


def get_matching_rule_beginning(text, text_rules, regex_rules, priority):
    answers = list()
    for idx in range(len(regex_rules)):
        r = regex_rules[idx]
        m = r.match(text)
        if m:
            if m.span()[0] == 0:
                answers.append(text_rules[idx])
    answers.sort(key=lambda x:[priority.index(y) for y in x])
    return answers
