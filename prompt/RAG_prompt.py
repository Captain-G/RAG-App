prompt = """
    ## Instructions
        Your task is to answer the following user question using the supplied search results.
        Always includes the key summary points at last.
            User Question: {user_question}
            Search Results: {search_text}
    """

legal_prompt = """
    ## Instructions
        Your task is to get very relevant information from legal documents.
        You need think for buyer & Give him very important points from documents that should be keep in mind. 
        You should include the negative points as caution at last.
        You should provide final score in positive sence. Means how many percentage you can see this documents are legit.
            User Question: {user_question}
            Search Results: {search_text}
    """

metadata_prompt = """
    Your task is to answer the following user question using the supplied search results. At the end of each search result will be Metadata. Cite the passages, their chunk index, and their URL in your answer.
    User Question: {user_question}
    Search Results: {search_text}
    """
