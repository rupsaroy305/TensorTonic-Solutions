def create_nsp_pairs(documents: list, pair_specs: list) -> list:
    result=[]
    for pair in pair_specs:
        doc_a=pair["doc_a"]
        sent_a=pair["sent_a"]
        doc_b=pair["doc_b"]
        sent_b=pair["sent_b"]
        sentence_a=documents[doc_a][sent_a]
        sentence_b=documents[doc_b][sent_b]
        is_next=int(doc_a==doc_b and sent_b==sent_a+1)
        result.append({"sentence_a": sentence_a,"sentence_b": sentence_b,"is_next": is_next})
    return result