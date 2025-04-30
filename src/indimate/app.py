import os
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


ollama_base_url = os.getenv("OLLAMA_BASE_URL")
llm_name = os.getenv("LLM", "llama2")


template = """
    Using your knowledge of physical crude oil trading, parse the folling fields

    'Counterparty:
    Grade:
    Incoterms:
    Bid or Offer:
    Pricing basis:
    Pricing premium:
    Pricing period:'

    exclusively using information from this text: '{text}'.

    The descriptions of the above are:

    Counterparty: parse which counterparty is buying/selling to us in the text. The counterparty is not the same as the grade.
    Use your knowledge of oil trading companies. If you cannot see the name within the text, leave this empty.
    Grade: which grade of crude oil is being indicated. Use your knowledge of crude oil grades.
    Incoterms: what are the incoterms of the deal.
    Bid or Offer: are they indicating a buy or sell. Usually words like 'bid' or 'offer' indicate this.
    Pricing basis: what is the floating basis of the deal.
    Pricing premium: what is the premium/discount over the basis, this is usually dated brent.
    Pricing period: what pricing period is being indicated?

    Give explanations of choices.
"""

model = OllamaLLM(model=llm_name, base_url=ollama_base_url)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model  # type: ignore


st.header("Read some indics in")


# Accept user questions/query
query = st.text_input("Add indics")

if query:
    res = chain.stream({"text": query})  # type: ignore

    text = ""

    if res:
        st.text(text)

        try:
            st.write_stream(res)
        except StopIteration:
            st.text("finished")
