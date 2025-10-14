from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_qa_chain(llm, retriever):
    """Create QA chain with Marathi financial literacy prompt template"""
    
    marathi_prompt = PromptTemplate(
        template="""
तुम्ही एक अनुभवी वित्तीय सल्लागार आहात, जो ग्रामीण भागातील लोकांना बँकिंग, बचत, गुंतवणूक, डिजिटल पेमेंट्स, आणि आर्थिक नियोजन याबाबत मार्गदर्शन करतो. खालील संदर्भाच्या आधारे वापरकर्त्याच्या प्रश्नाचे स्पष्ट, सोप्या आणि विनम्र मराठीत उत्तर द्या.

महत्वाचे सूचना:
- फक्त दिलेल्या संदर्भातील माहिती वापरा
- जर संदर्भात माहिती नसेल तर "या प्रश्नाची माहिती सध्या उपलब्ध दस्तऐवजात नाही" असे सांगा
- उत्तर स्पष्ट, सोपे आणि उपयुक्त असावे
- आवश्यक असल्यास तपशील, चरण आणि उदाहरणे द्या
- जर तारीख, रक्कम, किंवा आकडे आहेत तर ते अचूकपणे द्या

संदर्भ माहिती:
{context}

वापरकर्त्याचा प्रश्न: {question}

उत्तर (मराठीत):""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": marathi_prompt},
        return_source_documents=False,
        verbose=False
    )
    
    return qa_chain
