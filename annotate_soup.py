import spacy
from bs4 import BeautifulSoup

nlp = None


#def checkEqual(lst):
#    return not lst or lst.count(lst[0]) == len(lst)

def checkEqual(iterator):
    return len(set(iterator)) <= 1


def separate_common_parents(elems):
    iters = []
    for e in elems:
        #print([p.name for p in e.parents])
        iters.append(iter(reversed(list(e.parents))))
        #print(next(iters[-1]).name)

    i = 0
    equal = True
    while equal:
        d = None
        for it in iters:
            n_d = next(it, None)
            if n_d is None or (d is not None and d != n_d):
                equal = False
                break
        if equal:
            i += 1

    #ps_all = zip([reversed(list(e.parents)[:-1]) for e in elems])
    ##for ps_ in ps_all:
    ##    for ps in ps_:
    ##        print(str(ps))
    #i = 0
    #while i < len(ps_all) and checkEqual(ps_all[i]):
    #    i += 1
    common_parents = list(elems[0].parents)[-(i+1):-1]
    other_parents = []
    for e in elems:
        other_parents.append(list(e.parents)[:-(i+1)])
    return common_parents, other_parents


def text_element_to_annotated_tree(element, soup):
    offset = 0
    texts = []
    annots = []
    print('')
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
            parents = list(e.parents)[:-1]
            texts.append(t)
            annots.append(((offset, len(t)), e))
            #print('%i\t%i' % (len(e.string),len(t)))
            #print('%s:%i' % (t, len(t)))
            offset += len(t)

    text = u''.join(texts)
    #t2 = texts[0]
    #for i in range(len(texts) -1):
    #    t2.append(texts[i+1])
    #print(len(t2))
    if offset != len(text):
        print(sum([len(t) for t in texts]))
        raise Exception('wrong text size. offset=%i size=%i' % (offset, len(text)))

    doc = nlp(text)
    #for t in doc:
    #    print(t)
    text_doc = u''.join([t.text_with_ws for t in doc])
    print(sum([len(t) for t in texts]))
    print(sum([len(t.text_with_ws) for t in doc]))
    print(len(text))
    print(len(text_doc))
    print(len(doc.string))
    for i in range(len(text_doc)):
        if text_doc[i] != text[i]:
            print('%i: "%s" != "%s"' % (i, text[i], text_doc[i]))
    if text_doc != text:
        raise Exception('text after parsing is different: "%s" != "%s"' % (text, text_doc))
    # align tokens to text_annot elements. assume, token can belong to only one text_annot element!
    def end_pos((idx, l)):
        return idx+l
    token_start = 0
    parent_elements = {}
    #token_elements = {}
    #token_iter = doc.__iter__()
    #token = next(token_iter, None)
    token_ids = []
    annot_ids = []
    annot_id = 0
    for token in doc:
        token_ids.append(token.i)
        (elem_start, l), parents = annots[annot_id]
        while elem_start < token.idx + len(token.string):
            annot_ids.append(annot_id)
            annot_id += 1
            if annot_id == len(annots):
                #print(annot_id)
                break
            (elem_start, l), parents = annots[annot_id]

        #if len(annot_ids) > 0:
        token_end = token.idx + len(token.string)
        #elem_end = elem_start + l
        if len(annot_ids) > 0 and token_end >= end_pos(annots[annot_ids[-1]][0]):
            elem_end = end_pos(annots[annot_ids[-1]][0])
            if len(annot_ids) > 1 and len(token_ids) > 1:
                print('(%i:%i), (%i:%i) INSERT %s FOR %s' % (annots[annot_ids[0]][0][0], end_pos(annots[annot_ids[-1]][0]), doc[token_ids[0]].idx, doc[token_ids[-1]].idx + len(doc[token_ids[-1]].string) ,str([texts[a_id] for a_id in annot_ids]), str([doc[t_id].string for t_id in token_ids])))
            annot_elements = [annots[i][1] for i in annot_ids]
            common_parents, other_parents = separate_common_parents(annot_elements)
            common_parents = list(common_parents)
            for t_id in token_ids:
                tok = doc[t_id]
                parent_elements[t_id] = common_parents
                #print(tok)
            token_ids = []
            annot_ids = []

    #return
    #for i, ((elem_start, l), parents) in enumerate(annots):
    #    dummy = ('A\t', token_start, elem_start + l, unicode(token.string), texts[i], [p.name for p in parents])
    #    print(dummy)
    #    annot_ids.append(i)
    #    while token_start < elem_start + l and token is not None:
    #        # print('%i:%i,%i:%i:%s' % (i, elem_start, l, token_start, token))#

    #        token_ids.append(token.i)

    #        parent_elements[token.i] = parents
    #        #annot_ids = []
    #        token = next(token_iter, None)
    #        if token is None:
    #            break
    #        token_start = token.idx
    #        dummy = ('B\t', token_start, elem_start + l, unicode(token.string), texts[i], [p.name for p in parents])
    #        print(dummy)

    #    if len(token_ids) > 0:
    #        print('INSERT %s FOR %s' % (str(annot_ids), str(token_ids)))
    #        annot_ids = []
    #        #else:
    #        #    break
    #if len(doc) > 0 and max(parent_elements.keys()) != len(parent_elements.keys()) -1:
    #    print(text)

    new_elem = soup.new_tag(element.name, **element.attrs)
    for sent in doc.sents:
        def do_token(token):
            # add dependency edge and lexeme as classes
            elem_token = soup.new_tag('span', **{'class': 'spacy:dep/%s spacy:lex/%s' % (token.dep_, token.orth_)})
            # use token.string to get text content including whitespace
            elem_token.string = token.string
            str_ref = elem_token.string
            elem_ref = elem_token
            # add parents, eventually
            for p in parent_elements[token.i]:
                if token.i == token.head.i or p not in parent_elements[token.head.i]:
                    new_ = soup.new_tag(p.name, **p.attrs)
                    new_.append(elem_token)
                    elem_token = new_

            for child in token.children:
                # maintain original order
                if child.i < token.i:
                    str_ref.insert_before(do_token(child))
                else:
                    elem_ref.append(do_token(child))
            return elem_token

        new_elem.append(do_token(sent.root))
    return new_elem


def annotate_nlp(soup):
    text_block_elements = soup.find_all('div') + soup.find_all('p')
    # should not nest any other block element
    text_block_elements = [elem for elem in text_block_elements if elem.find_all('p') + elem.find_all('div') == []]
    inline_in_text_blocks = [elem for sublist in map(lambda x: x.find_all('span'), text_block_elements) for elem in
                             sublist]
    # should not already be contained in text_block_elements
    text_inline_elements = [elem for elem in soup.find_all('span') if elem not in inline_in_text_blocks]
    # remove spans whose parent gets already processed
    text_inline_elements = [elem for elem in text_inline_elements if len([p for p in elem.parents if p in text_inline_elements])==0]
    for elem in text_block_elements + text_inline_elements:
        if elem.find_all('p') + elem.find_all('div') == []:
            dummy = soup.new_tag('dummy')
            elem.insert_after(dummy)
            new_elem = text_element_to_annotated_tree(elem.extract(), soup)
            dummy.insert_after(new_elem)
            dummy.extract()


if __name__ == "__main__":
    print('parse html/xml ...')
    #contents = '<?xml version="1.0" encoding="iso-8859-1"?><!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><html xmlns="http://www.w3.org/1999/xhtml" class="client-js" dir="ltr" lang="en" xml:lang="en"><body class="mediawiki ltr sitedir-ltr mw-hide-empty-elt ns-0 ns-subject page-Physics rootpage-Physics skin-vector action-view"><div role="note" class="hatnote navigation-not-searchable">This article is about the field of science. For other uses, see <a href="/wiki/Physics_(disambiguation)" class="mw-disambig" title="Physics (disambiguation)"><span style="font-weight:bold;">Physics<span> </span>(disambiguation)</span></a>.</div></body></html>'
    #soup = BeautifulSoup(contents, 'html.parser')
    #with open('/home/arne/Downloads/en.wikipedia.org_wiki_Physics.html', 'r') as contents:
    with open('/home/arne/Downloads/test.html', 'r') as contents:
        soup = BeautifulSoup(contents, 'html.parser')
    # html_content = soup.find(class_='mw-parser-output')
    # text_elements = html_content.find_all('p') #+ html_content.find_all('span')

    print('load spacy ...')
    nlp = spacy.load('en')
    print('modify ...')
    annotate_nlp(soup)
    with open('/home/arne/Downloads/en.wikipedia.org_wiki_Physics.modified.html', "w") as file:
        file.write(str(soup))

    import xml.dom.minidom

    xml = xml.dom.minidom.parseString(str(soup))  # or xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = xml.toprettyxml()

    print(pretty_xml_as_string)
