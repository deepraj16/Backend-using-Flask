from flask import Flask, render_template, request, jsonify
import os
from chain_builder import create_qa_chain
from indexing import setup_llm
from loader import load_pdf_document
from vector_database import setup_embeddings_and_vectorstore

app = Flask(__name__)


qa_chain = None
document_loaded = False

def initialize_chatbot():
    """Initialize the chatbot with the fixed document"""
    global qa_chain, document_loaded
    
    try:
        
        chunks = load_pdf_document('addimsion.txt')
        
       
        print("Creating vector store...")
        vector_store = setup_embeddings_and_vectorstore(chunks)
        retriever = vector_store.as_retriever()
        
     
        llm = setup_llm()
        qa_chain = create_qa_chain(llm, retriever)
        
        document_loaded = True
       
        
    except Exception as e:

        raise

@app.route('/')
def home():
    """Render the main chat interface"""
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    global qa_chain, document_loaded
    
    try:
        if not document_loaded or qa_chain is None:
            return jsonify({
                'error': 'चॅटबॉट अजून तयार झालेला नाही. कृपया थोडा वेळ थांबा.'
            }), 500
        
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'कृपया प्रश्न लिहा'}), 400
        
        
        
    
        response = qa_chain.run(question)
        
   
        
        return jsonify({
            'answer': response,
            'status': 'success'
        })
        
    except Exception as e:

        return jsonify({
            'error': f'उत्तर देताना त्रुटी झाली: {str(e)}'
        }), 500

@app.route('/status')
def status():
    """Get current system status"""
    return jsonify({
        'document_loaded': document_loaded,
        'ready_for_chat': qa_chain is not None,
        'message': 'संजिवनी कॉलेज प्रवेश सल्लागार तयार आहे!' if document_loaded else 'चॅटबॉट लोड होत आहे...'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Marathi Chatbot'})

if __name__ == '__main__':
    initialize_chatbot()
    app.run(debug=True, host='0.0.0.0', port=5000)
    