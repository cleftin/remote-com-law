import os
from typing import List, Tuple, Any

# from dotenv import load_dotenv
from pydantic import ConfigDict
from pinecone import Pinecone

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda



# -----------------------------
# 1) 환경변수 로드
# -----------------------------
# load_dotenv()

# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GROUNDLINE_INDEX = os.getenv("GROUNDLINE_INDEX")
# BROADCOM_INDEX = os.getenv("BROADCOM_INDEX")
# PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# if not UPSTAGE_API_KEY:
#     raise ValueError("UPSTAGE_API_KEY가 없습니다.")
# if not PINECONE_API_KEY:
#     raise ValueError("PINECONE_API_KEY가 없습니다.")
# if not GROUNDLINE_INDEX or not BROADCOM_INDEX:
#     raise ValueError("GROUNDLINE_INDEX / BROADCOM_INDEX가 필요합니다.")


# # -----------------------------
# # 2) 임베딩 / LLM
# # -----------------------------
# # embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
# # llm = ChatUpstage()


# # -----------------------------
# # 3) Pinecone 인덱스 연결
# # -----------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)

# groundline_index = pc.Index(GROUNDLINE_INDEX)
# broadcom_index = pc.Index(BROADCOM_INDEX)

# groundline_db = PineconeVectorStore(
#     index=groundline_index,
#     embedding=embeddings,
#     namespace=PINECONE_NAMESPACE,
# )

# broadcom_db = PineconeVectorStore(
#     index=broadcom_index,
#     embedding=embeddings,
#     namespace=PINECONE_NAMESPACE,
# )


# -----------------------------
# 4) 멀티 인덱스 Retriever
# -----------------------------
class MultiPineconeRetriever(BaseRetriever):
    """
    두 개 이상의 PineconeVectorStore를 동시에 검색한 뒤,
    score 기준으로 정렬해서 상위 문서를 반환하는 커스텀 Retriever
    """

    vectorstores: List[Any]
    k_each: int = 5       # 각 인덱스에서 가져올 개수
    final_k: int = 5      # 최종 반환 개수

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs_with_scores: List[Tuple[Document, float]] = []

        for vs in self.vectorstores:
            # PineconeVectorStore의 similarity_search_with_score 사용
            # score는 vector store 구현에 따라 거리/유사도 의미가 다를 수 있으므로
            # 실제 결과를 보고 정렬 방향을 확인하는 것이 안전함
            docs_with_scores = vs.similarity_search_with_score(query, k=self.k_each)
            all_docs_with_scores.extend(docs_with_scores)

        # 점수 정렬
        # 현재 사용 코드 흐름에 맞춰 reverse=True 유지
        # 만약 결과가 이상하면 reverse=False로 바꿔 테스트하세요.
        all_docs_with_scores = sorted(
            all_docs_with_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        # 최종 상위 문서 추출
        top_docs = []
        for rank, (doc, score) in enumerate(all_docs_with_scores[: self.final_k], start=1):
            # 출처 확인용 metadata 추가
            doc.metadata["retrieval_score"] = score
            doc.metadata["retrieval_rank"] = rank
            top_docs.append(doc)
            print("doc\n\n",doc.page_content)

        return top_docs


# multi_retriever = MultiPineconeRetriever(
#     vectorstores=[groundline_db, broadcom_db],
#     k_each=5,
#     final_k=5,
# )


def format_docs(docs: List[Document]) -> str:

    formatted_docs = []

    for doc in docs:

        regulation = doc.metadata.get("regulation_name", "")
        article = doc.metadata.get("article", "")
        # clause = doc.metadata.get("clause", "")
        # source = doc.metadata.get("source", "")

        text = f"""
규정명 : {regulation}
조문: {article} 
본문:
{doc.page_content}
"""

        formatted_docs.append(text)

    return "\n\n".join(formatted_docs)


def extract_metadata(data):
    docs = data["docs"]
    question = data["input"]

    if not docs:
        return {
            "context": "",
            "regulation": "관련 규정 또는 기술기준 확인 불가",
            "article": "",
            "input": question,
        }

    top_doc = docs[0]

    return {
        "context": format_docs(docs),
        "regulation": top_doc.metadata.get("regulation_name", "관련 규정 또는 기술기준 확인 불가"),
        "article": top_doc.metadata.get("article", ""),
        "input": question,
    }

# # -----------------------------
# # 5) QA 프롬프트
# # -----------------------------
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
# 너는 통신설비 기술 질의응답 전문가다.
# 반드시 제공된 context만 근거로 답변하라.
# context에 없으면 추측하지 말고 "관련 규정 또는 기술기준 확인 불가, 제공된 문서에서 확인되지 않습니다."라고 답하라.
# 답변은 한국어로, 핵심부터 간결하게 작성하라.

# 반드시 아래 형식으로만 답하라.

# {regulation} {article}, 답변내용

# 예시:
# 접지설비·구내통신설비·선로설비 및 통신공동구등에 대한 기술기준 제8조, 통신설비의 접지저항은 10Ω 이하로 해야 합니다.

# <context>
# {context}
# </context>
# """.strip(),
#         ),
#         ("human", "{input}"),
#     ]
# )


# rag_chain = (
#     {
#         "docs": multi_retriever,
#         "input": RunnablePassthrough(),
#     }
#     | RunnableLambda(extract_metadata)
#     | qa_prompt
#     | llm
# )



######################################################################

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm(model='upstage'):
    if model == 'upstage':
        llm = ChatUpstage()
    else:
        raise ValueError(f"Invalid model: {model}")
   
    return llm


# def get_dictionary_chain(user_message):

#     llm = get_llm()

#     dictionary = ["사람을 나타내는 표현 -> 거주자"]

#     prechange_prompt = ChatPromptTemplate.from_template(f"""
#     사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
#     만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
#     사전 : {dictionary} 
#     질문 : {{question}}
#     """)

#     dictionary_chain = prechange_prompt | llm | StrOutputParser()  # Extracts string from AIMessage

#     new_question = dictionary_chain.invoke({"question":user_message})

#     return new_question


def get_retrieved_docs(user_message):
    
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
   
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROUNDLINE_INDEX = os.getenv("GROUNDLINE_INDEX")
    BROADCOM_INDEX = os.getenv("BROADCOM_INDEX")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

    if not UPSTAGE_API_KEY:
        raise ValueError("UPSTAGE_API_KEY가 없습니다.")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY가 없습니다.")
    if not GROUNDLINE_INDEX or not BROADCOM_INDEX:
        raise ValueError("GROUNDLINE_INDEX / BROADCOM_INDEX가 필요합니다.")

    # -----------------------------
    # 3) Pinecone 인덱스 연결
    # -----------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY)

    groundline_index = pc.Index(GROUNDLINE_INDEX)
    broadcom_index = pc.Index(BROADCOM_INDEX)

    groundline_db = PineconeVectorStore(
        index=groundline_index,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    broadcom_db = PineconeVectorStore(
        index=broadcom_index,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    multi_retriever = MultiPineconeRetriever(
        vectorstores=[groundline_db, broadcom_db],
        k_each=5,
        final_k=5,
    )

    return multi_retriever






    index_name = 'tax-index'
    # pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    # pc = Pinecone(api_key=pinecone_api_key)
    # print("PINECONE_API_KEY =", os.environ.get("PINECONE_API_KEY"))
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

    new_question = get_dictionary_chain(user_message)

    retrieved_docs = database.similarity_search(new_question, k=3)

    return retrieved_docs


def get_ai_message(user_message):
    llm = get_llm()
    retrieved_docs = get_retrieved_docs(user_message)
    # new_question = get_dictionary_chain(user_message)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retrieved_docs, contextualize_q_prompt
    )


    # -----------------------------
    # 5) QA 프롬프트
    # -----------------------------
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    너는 통신설비 기술 질의응답 전문가다.
    반드시 제공된 context만 근거로 답변하라.
    context에 없으면 추측하지 말고 "관련 규정 또는 기술기준 확인 불가, 제공된 문서에서 확인되지 않습니다."라고 답하라.
    답변은 한국어로, 핵심부터 간결하게 작성하라.

    반드시 아래 형식으로만 답하라.

    {regulation} {article}, 답변내용

    예시:
    접지설비·구내통신설비·선로설비 및 통신공동구등에 대한 기술기준 제8조, 통신설비의 접지저항은 10Ω 이하로 해야 합니다.

    <context>
    {context}
    </context>
    """.strip(),
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    rag_chain = (
        {
            "docs": history_aware_retriever,
            "input": RunnablePassthrough(),
        }
        | RunnableLambda(extract_metadata)
        | qa_prompt
        | llm
    )

    ai_message = rag_chain(user_message)

    return ai_message.content
    # system_prompt = (
    #     "you are an assistant for question-answering tasks."
    #     "use the following pieces of retrieved context to answer"
    #     "the question. If you don't know the answer, just say that you don't know. "
    #     "use three sentences maximum and keep the answer concise."
    #     "\n\n"
    #     "{context}"
    # )
    # qa_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),      
    #     ]
    # )
    # question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )



    # prompt = f"""[Identity]
    # - 당신은 최고의 한국 소득세 전문가입니다
    # - [Context]를 참고해서 사용자의 질문에 답변해주세요

    # [Context]
    # {retrieved_docs}
    # Question : {new_question}
    # """

    # tax_chain = {"input": dictionary_chain} | conversational_rag_chain
    # ai_message = tax_chain.invoke(
    #     {"question": user_message},
    #     config={
    #         "configurable": {
    #             "session_id": "abc123"
    #         }
    #     }
    #     )

    #  ai_message = llm.invoke(prompt)
    # return ai_message.content