import pandas as pd

from regex_util import get_best_rules_for_all
from regex_util import get_matching_rule_beginning
from regex_util import get_corresponding_rule_regex_text
from missing_chapter_util import find_missing_chapters
from missing_chapter_util import convert_to_int

import itertools
import re

import pickle
import lxml.etree as ET

import copy


def group_ranges(L):
    """
    Collapses a list of integers into a list of the start and end of
    consecutive runs of numbers. Returns a generator of generators.
    >>> [list(x) for x in group_ranges([1, 2, 3, 5, 6, 8])]
    [[1, 3], [5, 6], [8]]
    """
    for w, z in itertools.groupby(L, lambda x, y=itertools.count(): next(y)-x):
        grouped = list(z)
        yield (x for x in [grouped[0], grouped[-1]][:len(grouped)])

def header_to_xml(header_lines, book, output_xml_path):
    
    header_lines = [[y for y in x] for x in group_ranges(header_lines)]
    
    # First line: last line
    d = {x[0]:x[-1] for x in header_lines}

    # Delete section tags
    ET.strip_tags(book, "section")

    for from_line, to_line in d.items():
        f = book.find('.//line[@num="' + str(from_line) + '"]')

        new_element = ET.Element('header')

        prev = f.getprevious()
        if prev is not None:
            for line_num in range(from_line, to_line + 1):
                e = book.find('.//line[@num="' + str(line_num) + '"]')
                new_element.append(e)
            prev.addnext(new_element)
        else:
            parent = f.getparent()
            for line_num in range(from_line, to_line + 1):
                e = book.find('.//line[@num="' + str(line_num) + '"]')
                new_element.append(e)
            parent.insert(0, new_element)


    ET.strip_tags(book, "line")

    # Write to file
    with open(output_xml_path, 'wb') as f:
        f.write(ET.tostring(book, pretty_print=True))
            
            


def get_high_vicinity_indices(very_high_threshold, very_low_threshold, new_probs, body_start):
    
    # Find indices with probability above very high threshold
    pos_pred_indices = [idx for idx in range(len(new_probs)) if new_probs[idx] > very_high_threshold and idx >= body_start]
    
    mod_pos_pred_groups = list()
    seen = set()
    
    for index in pos_pred_indices:
        # If index is already seen, ignore
        if index in seen:
            continue
        # Find consecutive indices that come after this index, and have probability > very_low_threshold
        # (Expand to the right)
        idx = index
        new_group = [idx]
        seen.add(idx)
        idx += 1
        while idx < len(new_probs) and new_probs[idx] > very_low_threshold:
            new_group.append(idx)
            seen.add(idx)
            idx += 1
        # Append to list of all indices
        mod_pos_pred_groups.append(new_group)
    
    return mod_pos_pred_groups


def merge_close_groups(list_of_groups, tokens):
    new_list_of_groups = list()
    idx = 0
    while idx < len(list_of_groups) - 1:
        if abs(max(list_of_groups[idx]) - min(list_of_groups[idx + 1])) < 20 and '[unused1]' not in tokens[max(list_of_groups[idx]) + 1:min(list_of_groups[idx + 1])]:
            start = min(list_of_groups[idx])
            end = max(list_of_groups[idx + 1])
            curr = [x for x in range(start, end + 1)]
            new_list_of_groups.append(curr)
            idx += 1
        else:
            new_list_of_groups.append(list_of_groups[idx])
        idx += 1
    if idx == len(list_of_groups) - 1:
        new_list_of_groups.append(list_of_groups[idx])
    
    if new_list_of_groups == list_of_groups:
        return new_list_of_groups
    
    return merge_close_groups(new_list_of_groups, tokens)
    

    
def get_header_attrs(text, rule):
    #print([text], rule)
    desc = None
    number = None
    number_text = None
    number_type = None
    title = None
    rule_text = ','.join(rule)
    
    number_list = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST', 'numeral']
    
    curr_text = text
    for element in rule:
        if len(element) == 0:
            break
        
        if element == 'whitespace':
            curr_text = curr_text.lstrip()
            continue
        
        r = re.compile(get_corresponding_rule_regex_text(element))
        m = r.match(curr_text)
        start, end = m.span()
        
        if element == 'desc':
            desc = curr_text[start:end].strip()
        elif element in number_list:
            number = convert_to_int(curr_text[start:end].strip(), element)
            number_text = curr_text[start:end].strip()
            number_type = element
        elif element in ['title_upper', 'title_lower']:
            title = curr_text[start:end].strip()
        curr_text = curr_text[end:]
    #print('Returning ', (desc, number, number_text, number_type, title))
    return desc, number, number_text, number_type, title, rule_text


    
def get_header_lines(df, all_seqs, all_seqs_orig, rules, priority, body_start, output_pickle_prefix, book):
    tokens = [str(x) for x in list(df['token'])]
    probs = list(df['prob'])
    line_nums = list(df['line_number'])
    
    body_start_word = line_nums.index(body_start)
    
    # Get indices in vicinity of high probability indices
    
    # Top 10% tokens
    top_10_percent = len(df) // 10
    very_high_threshold = sorted(df.loc[body_start_word:]['prob'])[-top_10_percent]
    
    
    
    very_low_threshold = 0.1
    pos_pred_groups = get_high_vicinity_indices(very_high_threshold, very_low_threshold, probs, body_start_word)
    
    # Merge groups close to each other
    mod_pos_pred_groups = merge_close_groups(pos_pred_groups, tokens)
    
    # Replace the [unused1] token with \n and strip
    stripped_texts = [' '.join([tokens[idx].replace('[unused1]', '\n') for idx in x]).strip() for x in mod_pos_pred_groups]
    
    # Keep only those groups which have at least one alphanumeric character when stripped
    mod_pos_pred_groups = [mod_pos_pred_groups[idx] for idx in range(len(mod_pos_pred_groups)) if any([x.isalnum() for x in stripped_texts[idx]])]
    
    
    # Remove newlines from beginning
    mod_pos_pred_groups_2 = list()
    for group in mod_pos_pred_groups:
        first_idx = 0
        while first_idx < len(group) and tokens[group[first_idx]] == '[unused1]':
            first_idx += 1
        mod_pos_pred_groups_2.append(group[first_idx:])
    mod_pos_pred_groups = mod_pos_pred_groups_2
    
    # Extend to entire previous and next part of line
    mod_pos_pred_groups_2 = list()
    for group in mod_pos_pred_groups:
        tmp = group
        # Previous
        first_idx = group[0] - 1
        line = line_nums[group[0]]
        while line_nums[first_idx] == line:
            tmp.insert(0, first_idx)
            first_idx -= 1
        # Next
        first_idx = group[-1] + 1
        line = line_nums[group[-1]]
        while first_idx < len(line_nums) and line_nums[first_idx] == line:
            tmp.append(first_idx)
            first_idx += 1
        mod_pos_pred_groups_2.append(tmp)
    mod_pos_pred_groups = mod_pos_pred_groups_2
    
    # Add newlines at the end if present (Adding for title_upper to match)
    mod_pos_pred_groups_2 = list()
    for group in mod_pos_pred_groups:
        new_group = group
        last_idx = group[-1] + 1
        while last_idx < len(tokens) and tokens[last_idx] == '[unused1]':
            new_group.append(last_idx)
            last_idx += 1
        mod_pos_pred_groups_2.append(new_group)
    mod_pos_pred_groups = mod_pos_pred_groups_2
    
    # Convert groups to texts for regex matching
    likely_headers = [' '.join([tokens[idx].replace('[unused1]', '\n') for idx in x]) for x in mod_pos_pred_groups]
    
    header_lines = set()
    for g in mod_pos_pred_groups:
        for x in g:
            header_lines.add(line_nums[x])
    header_lines = sorted(list(header_lines))
    header_to_xml(header_lines, copy.deepcopy(book), output_pickle_prefix + '_stage1_headers.xml')
    
    # Find the best matching rule for each header
    rules_found = get_best_rules_for_all(likely_headers, all_seqs, rules, priority)
    
    
    # Look for missing chapters using each rule
    rules_found = [[x for x in r if x] if r else [] for r in rules_found]
    num_set = set(['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST'])
    clipped_rules_found = list()
    for r in rules_found:
        if len(r) == 0:
            clipped_rules_found.append(r)
            continue
        tmp = list()
        tmp_idx = 0
        while tmp_idx < len(r):
            tmp.append(r[tmp_idx])
            if r[tmp_idx] in num_set:
                break
            tmp_idx += 1
        clipped_rules_found.append(tmp)
        
    d_rules = dict()
    for r_idx, r in enumerate(clipped_rules_found):
        if len(r) == 0:
            continue
        t = tuple(r)
        if t not in d_rules:
            d_rules[t] = [r_idx]
        else:
            d_rules[t].append(r_idx)
    
    d_rules_new = dict()
    for t in d_rules:
        r = list(t)
        d_rules_new[t] = find_missing_chapters(df, [mod_pos_pred_groups[r_idx] for r_idx in d_rules[t]], r, body_start_word)
    
    # Improve each match to include title if present
    valid_indices = [idx for idx in range(len(all_seqs_orig))]
    new_seqs = [all_seqs[idx] for idx in valid_indices]
    new_rules = [rules[idx] for idx in valid_indices]
    
    ans = list()
    attrs_dict = dict()
    
    all_header_groups = list()
    for r in d_rules_new.keys():
        if list(r) == ['title_upper']:
            continue
        rule_groups = d_rules_new[r]
        all_header_groups += rule_groups
    all_header_groups = sorted(all_header_groups)
    
    next_group_index = dict()
    for idx in range(len(all_header_groups) - 1):
        next_group_index[all_header_groups[idx][0]] = all_header_groups[idx + 1][0]
    
    sorted_rule_list = list(d_rules_new.keys())
    sorted_rule_list.sort(key=lambda x:[priority.index(y) for y in x])
    seen = set()
    
    for rule in sorted_rule_list:
        for group_idx, group in enumerate(d_rules_new[rule]):
            start_idx = group[0]
            
            end_idx = start_idx + 100
            if start_idx in next_group_index:
                end_idx = min(end_idx, next_group_index[start_idx])
            
            tmp_text = ' '.join([x if x != '[unused1]' else '\n' for x in tokens[start_idx:end_idx]])
            matched_rules = get_matching_rule_beginning(tmp_text, new_seqs, new_rules, priority)
            if len(matched_rules) > 0:
                matched_bool = False
                for r_orig in matched_rules:
                    try:
                        r = new_rules[new_seqs.index(r_orig)]
                        m = r.match(tmp_text)
                        start, end = m.span()
                        match_len = end - start
                        if tmp_text[start:end].strip(' ')[-1] != '\n' and tmp_text[end:].strip(' ')[0] != '\n':
                            continue

                        attrs = get_header_attrs(tmp_text[start:end], r_orig)
                        matched_bool = True
                        break
                    except:
                        pass
                if not matched_bool:
                    continue
                tmp = list()
                curr_str = ''
                
                new_iter = start_idx
                while new_iter < len(tokens):
                    curr_str += (tokens[new_iter] if tokens[new_iter] != '[unused1]' else '\n')
                    if len(curr_str) > match_len:
                        break
                    tmp.append(new_iter)
                    curr_str += ' '
                    new_iter += 1
                if tmp:
                    if not any([index in seen for index in tmp]):
                        ans.append(tmp)
                        seen.update(tmp)
                        attrs_dict[line_nums[start_idx]] = attrs
    
    
    
    line_nos = set()
    for group in ans:
        for idx in group:
            line_nos.add(line_nums[idx])
            
    final_ans = list()
    line_nos_sorted = sorted(list(line_nos))
    
    idx = 0
    while idx < len(line_nos_sorted):
        if line_nos_sorted[idx] in attrs_dict:
            tmp = list()
            tmp.append(line_nos_sorted[idx])
            inner_idx = idx + 1
            while inner_idx < len(line_nos_sorted) and line_nos_sorted[inner_idx] not in attrs_dict:
                tmp.append(line_nos_sorted[inner_idx])
                inner_idx += 1
            final_ans.append((tmp, attrs_dict[line_nos_sorted[idx]]))
            
            idx = inner_idx
            continue
            
        idx += 1
    
    
    header_lines = set()
    for g, _ in final_ans:
        for x in g:
            header_lines.add(x)
    header_lines = sorted(list(header_lines))
    
    header_to_xml(header_lines, copy.deepcopy(book), output_pickle_prefix + '_stage2_headers.xml')
    
    header_lines = final_ans
    
    
    
    # Keep only those groups which do not start with a I'm, I've, I'd, I'll
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’'
    stripped_texts = [' '.join([tokens[idx].replace('[unused1]', '\n') for idx in x[0]]).strip() for x in header_lines]
    
    header_lines_new = list()
    for idx in range(len(stripped_texts)):
        if not any([stripped_texts[idx].lstrip().lstrip(punctuation).lstrip().startswith(x) for x in {"I'm", "I've", "I’m", "I’ve", "I'd", "I’d", "I'll", "I’ll"}]):
            header_lines_new.append(header_lines[idx])
    
    header_lines = header_lines_new
    
    # Remove false positives
    remove_indices = set()
    for idx, hl in enumerate(header_lines):
        line_nums, attrs = hl
        desc, number, number_text, number_type, title, rule_text = attrs
        if number == None:
            continue
        
        i = idx + 1
        while i < len(header_lines) and header_lines[i][1][1] is None:
            i += 1
        if i < len(header_lines) and header_lines[i][1][1] == number + 1 and header_lines[i][1][3] == number_type:
            for i2 in range(idx + 1, i):
                remove_indices.add(i2)
    final_ans = [elem for idx, elem in enumerate(header_lines) if idx not in remove_indices]
    
    
    header_lines = set()
    for g, _ in final_ans:
        for x in g:
            header_lines.add(x)
    header_lines = sorted(list(header_lines))
    header_to_xml(header_lines, copy.deepcopy(book), output_pickle_prefix + '_stage3_headers.xml')
    
    return final_ans
