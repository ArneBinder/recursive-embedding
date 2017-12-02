import spacy
from bs4 import BeautifulSoup

nlp = None


#def checkEqual(lst):
#    return not lst or lst.count(lst[0]) == len(lst)

def checkEqual(iterator):
    return len(set(iterator)) <= 1


def get_common_parents(parents):
    iters = []
    for ps in parents:
        iters.append(iter(reversed(list(ps))))

    i = 0
    equal = True
    while equal:
        d = None
        for it in iters:
            n_d = next(it, None)
            if n_d is None or (d is not None and d != n_d):
                equal = False
                break
            d = n_d
        if equal:
            i += 1

    common_parents = list(parents[0])[-i]
    return common_parents


def separate_common_parents(elems):
    common_parents = get_common_parents([list(e.parents) for e in elems])
    i = len(common_parents)
    other_parents = []
    for e in elems:
        other_parents.append(list(e.parents)[:-(i+1)])
    return common_parents, other_parents


def handle_ids(token_ids, annot_ids, annots, parent_elements, doc):
    # elem_end = end_pos(annots[annot_ids[-1]][0])
    # if len(annot_ids) > 1 and len(token_ids) > 1:
    #    print('(%i:%i), (%i:%i) INSERT %s FOR %s' % (annots[annot_ids[0]][0][0], end_pos(annots[annot_ids[-1]][0]), doc[token_ids[0]].idx, doc[token_ids[-1]].idx + len(doc[token_ids[-1]].string) ,str([texts[a_id] for a_id in annot_ids]), str([doc[t_id].string for t_id in token_ids])))
    annot_elements = [annots[i][1] for i in annot_ids]
    common_parents = get_common_parents([list(e.parents) for e in annot_elements])
    # common_parents = list(common_parents)
    for t_id in token_ids:
        # tok = doc[t_id]
        parent_elements[t_id] = common_parents
        #p = common_parents
        #p_tags = []
        #while p.parent is not None:
        #    p_tags.append(p.name)
        #    p = p.parent
        #print('%i:\t%s "%s"' % (t_id, str(list(reversed(p_tags))), doc[t_id].string))


def text_element_to_annotated_tree(element, soup, nlp):
    if element.text.strip() == '':
        return element

    offset = 0
    texts = []
    annots = []
    #print('')
    for e in element.next_elements:
        if e.string is not None and e.name is None:# and e.string.strip() != '':
            # add text elements containing only whitespace to previous one
            #if e.string.strip() == '' and len(texts) > 0:
            #    texts[-1] = texts[-1] + e.string
            #    (o, l), ps = annots[-1]
            #    annots[-1] = (o, len(texts[-1])), ps
            #    offset += len(e.string)
            #    continue

            #print(len(e.string))
            t = e.string
            #if type(t) == str:
            t = unicode(t)
            # hack to fix spacy "bug": spacy replaces newlines by spaces
            t = t.replace('\n', ' ')
            #parents = list(e.parents)[:-1]
            texts.append(t)
            annots.append((offset, e))
            #print('%i\t%i' % (len(e.string),len(t)))
            #print('%s:%i' % (t, len(t)))
            offset += len(t)

    text = u''.join(texts)
    #t2 = texts[0]
    #for i in range(len(texts) -1):
    #    t2.append(texts[i+1])
    #print(len(t2))
    if offset != len(text):
        #print(sum([len(t) for t in texts]))
        raise Exception('wrong text size. offset=%i size=%i' % (offset, len(text)))

    doc = nlp(text)
    #for t in doc:
    #    print(t)
    text_doc = u''.join([t.text_with_ws for t in doc])
    #print(sum([len(t) for t in texts]))
    #print(sum([len(t.text_with_ws) for t in doc]))
    #print('"%s"' % text)
    #print(len(text))
    #print('"%s"' % text_doc)
    #print(len(text_doc))
    #print(len(doc.string))
    for i in range(min(len(text_doc), len(text))):
        if text_doc[i] != text[i]:
            raise Exception('%i: "%s" != "%s"' % (i, text[i], text_doc[i]))
    # TODO: fix hack! (trailing whitespace removed by spacy)
    if text_doc[:min(len(text_doc), len(text))] != text[:min(len(text_doc), len(text))]:
        raise Exception('text after parsing is different: "%s" != "%s"' % (text, text_doc))

    # align tokens to annots elements. assume, token can belong to only one text_annot element!
    def end_pos((idx, l)):
        return idx+l

    parent_elements = {}
    token_ids = []
    annot_ids = []
    annot_id = 0
    # don't add leading whitespace elements
    #add = False
    for token in doc:
        #if token.pos_ == 'SPACE':
        #    continue
        token_ids.append(token.i)
        elem_start, _ = annots[annot_id]
        # while elem_start < token.idx + len(token.string.rstrip()):
        while elem_start < token.idx + len(token):
            # if one none whitespace element is found, add remaining, even if they are whitespace elements
            #if texts[annot_id].strip() != '':
            #    add = True
            #if add:
            annot_ids.append(annot_id)
            if annot_id == len(annots) - 1:
                #print(annot_id)
                break
            annot_id += 1
            elem_start, _ = annots[annot_id]

        #if len(annot_ids) > 0:
        token_end = token.idx + len(token.string)
        #elem_end = elem_start + l

        # dont handle ids, if annots are exceeded, but tokens not
        #  end_pos(annots[annot_ids[-1]][0]) and \
        if len(annot_ids) > 0 and token_end >= annots[annot_ids[-1]][0] + len(texts[annot_ids[-1]]) and \
                not (annot_id == len(annots) - 1 and token.i < len(doc) - 1):
            handle_ids(token_ids, annot_ids, annots, parent_elements, doc)
            token_ids = []
            annot_ids = []
            #add = False

    # handle remaining token_ids/annot_ids
    if len(annot_ids) > 0:
        if len(token_ids) == 0:
            raise Exception('empty token_ids, but annot_ids contains elements')
        handle_ids(annot_ids, token_ids, annots, parent_elements, doc)
        token_ids = []
        annot_ids = []

    recreated_parents = {}
    new_elem = soup.new_tag(element.name, **element.attrs)
    recreated_parents[element] = new_elem
    #prev_root = None
    for sent in doc.sents:
        def do_token(token):
            # add dependency edge and lexeme as classes
            elem_token = soup.new_tag('span', **{'class': 'spacy:dep/%s spacy:lex/%s' % (token.dep_, token.orth_)})
            # use token.string to get text content including whitespace
            elem_token.string = token.string
            ref_str = elem_token.string
            ref_token = elem_token
            # add parents, eventually
            #if token.i not in parent_elements:
            #    return None
            p = parent_elements[token.i]
            #for p in parent_elements[token.i]:
            while p not in recreated_parents:
                #if p not in root_parents and (token.i == token.head.i or p not in parent_elements[token.head.i]):
                #if p not in recreated_parents:
                new_parent = soup.new_tag(p.name, **p.attrs)
                recreated_parents[p] = new_parent
                new_parent.append(elem_token)
                elem_token = new_parent
                #else:
                #    break
                p = p.parent

            if token.head.i == token.i:
                recreated_parents[p].append(elem_token)
                elem_token = recreated_parents[p]

            last_child_after = ref_str
            for child in token.children:
                c_token = do_token(child)
                if c_token is not None:
                    # maintain original order
                    if child.i < token.i:
                        ref_str.insert_before(c_token)
                    else:
                        last_child_after.insert_after(c_token)
                        last_child_after = c_token
            return elem_token

        #sent_parents = [parent_elements[t.i] for t in sent]
        #root_parents = get_common_parents(sent_parents)

        do_token(sent.root)
        #print(sent.string)
        #prev_root = sent.root
    return new_elem


def annotate_nlp(soup, nlp):
    text_block_tags = ['div', 'p', 'li']
    #text_block_tags = ['p']
    #block_tags = ['adress', 'article', 'aside', 'ausio', 'video', 'blockquote', 'canvas', 'dd', 'div', 'dl', 'fieldset',
    #              'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hgroup',
    #              'hr', 'noscript', 'ol', 'ul', 'output', 'p', 'pre', 'section', 'table', 'tfoot', 'ul']
    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'address', 'blockquote', 'dl', 'div',
                  'fieldset', 'form', 'hr', 'noscript', 'table']
    #inline_tags = ['b', 'big', 'i', 'small', 'tt', 'abbr', 'acronym', 'cite', 'code', 'dfn', 'em', 'kbd', 'strong',
    #               'samp', 'var', 'a', 'bdo', 'br', 'img', 'map', 'object', 'q', 'script', 'span', 'sub', 'sup',
    #               'button', 'input', 'label', 'select', 'textarea']
    inline_tags = ['span', 'textarea', 'label']
    # add list item tag
    #inline_tags.append('li')
    # should not nest any other block element
    text_block_elements = [elem for elem in soup.find_all(text_block_tags) if elem.find_all(set(block_tags + text_block_tags)) == []]
    inline_in_text_blocks = [elem for sublist in map(lambda x: x.find_all(inline_tags), text_block_elements) for elem in
                             sublist]
    # should not already be contained in text_block_elements
    text_inline_elements = [elem for elem in soup.find_all(inline_tags) if elem not in inline_in_text_blocks]
    # remove spans whose parent gets already processed
    text_inline_elements = [elem for elem in text_inline_elements if len([p for p in elem.parents if p in text_inline_elements])==0]
    print('elements to modify: %i' % len(text_block_elements + text_inline_elements))
    for i, elem in enumerate(text_block_elements + text_inline_elements):
        #print(repr(elem))
        # append dummy element as marker to re-insert the modified one later at teh correct position
        dummy = soup.new_tag('dummy')
        elem.insert_after(dummy)
        elem_backup = elem.extract()
        try:
            new_elem = text_element_to_annotated_tree(elem_backup, soup, nlp)
        except Exception as e:
        #    #print('%i could not modify element: %s' % (i, e))
            print('%i: element failed, restore:: %s' % (i, repr(elem)))
            new_elem = elem_backup
        #    #print(repr(elem))
        dummy.insert_after(new_elem)
        dummy.extract()


if __name__ == "__main__":
    print('parse html/xml ...')
    #contents = '<?xml version="1.0" encoding="iso-8859-1"?><!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><html xmlns="http://www.w3.org/1999/xhtml" class="client-js" dir="ltr" lang="en" xml:lang="en"><body class="mediawiki ltr sitedir-ltr mw-hide-empty-elt ns-0 ns-subject page-Physics rootpage-Physics skin-vector action-view"><div role="note" class="hatnote navigation-not-searchable">This article is about the field of science. For other uses, see <a href="/wiki/Physics_(disambiguation)" class="mw-disambig" title="Physics (disambiguation)"><span style="font-weight:bold;">Physics<span> </span>(disambiguation)</span></a>.</div></body></html>'
    #soup = BeautifulSoup(contents, 'html.parser')
    #with open('/home/arne/Downloads/en.wikipedia.org_wiki_Physics.html', 'r') as contents:
    fn = 'data/en.wikipedia.org_wiki_Physics.html'
    with open(fn, 'r') as contents:
    #with open('/home/arne/Downloads/test.html', 'r') as contents:
        soup = BeautifulSoup(contents, 'html.parser')
    # html_content = soup.find(class_='mw-parser-output')
    # text_elements = html_content.find_all('p') #+ html_content.find_all('span')

    print('load spacy ...')
    nlp = spacy.load('en')
    print('modify ...')
    annotate_nlp(soup, nlp)
    with open('%s.nlp.html' % fn, "w") as file:
        file.write(str(soup))

    import xml.dom.minidom

    xml = xml.dom.minidom.parseString(str(soup))  # or xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = xml.toprettyxml()

    #print(pretty_xml_as_string)
