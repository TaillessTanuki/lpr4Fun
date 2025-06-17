import re


valid_prefixes = {
    'A','B','D','E','F','G','H','K','L','M','N','P','R','S','T','W','Z',
    'AA','AB','AD','AE','AG','BA','BB','BD','BE','BG','BH','BK','BL','BM','BN','BP',
    'DA','DB','DC','DD','DE','DG','DH','DK','DL','DM','DN','DP','DR','DT','DW',
    'EA','EB','ED','KB','KH','KT','KU','PA','PB'
}

ocr_corrections = {
    '6': 'G',
    '8': 'B',
    '0': 'D',
    '1': 'I',
    '5': 'S'
}


def correct_prefix(text):
    # Try 2-letter prefix first, then 1-letter
    for i in [2, 1]:
        prefix = text[:i]
        if prefix in valid_prefixes:
            return text  # Already valid
        # Try correction
        corrected = ''.join(ocr_corrections.get(c, c) for c in prefix)
        if corrected in valid_prefixes:
            return corrected + text[i:]
    return text  # No valid correction

def extract_plate_core(text):
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Match prefix letters (1–2), digits (at least 1), and optional suffix letters
    match = re.match(r'^([A-Z]{1,2})(\d+)([A-Z]*)', clean)
    if match:
        prefix, digits, suffix = match.groups()

        # Limit to 4 digits only
        digits = digits[:4]
        suffix = suffix[:3]  # Limit to 3 letters
        return prefix + digits + suffix

    return ''
