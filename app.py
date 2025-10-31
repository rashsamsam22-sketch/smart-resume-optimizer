from typing import Any, Dict
from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from langchain.text_splitter import CharacterTextSplitter
from flask_cors import CORS
from dotenv import load_dotenv
import fitz
import json
import traceback
from io import BytesIO
import re

load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# ============================================================================
# AGENTIC AI SETUP - LAZY LOADING
# ============================================================================

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

# LAZY LOAD EMBEDDINGS - Only load when needed
embeddings = None

def get_embeddings():
    """Lazy load embeddings to save memory on startup"""
    global embeddings
    if embeddings is None:
        print("📦 Loading embeddings model (first time only)...")
        from sentence_transformers import SentenceTransformer
        # Use lightweight model: 22MB instead of 100MB+
        embeddings = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        print("✅ Embeddings loaded")
    return embeddings

# LAZY LOAD LLM
llm = None

def get_llm():
    """Lazy load LLM to avoid startup timeout"""
    global llm
    if llm is None:
        print("🤖 Initializing LLM (first request)...")
        from langchain_cerebras import ChatCerebras
        llm = ChatCerebras(
            api_key=CEREBRAS_API_KEY,
            model="llama3.1-8b",
            temperature=0.3,
            max_tokens=800
        )
        print("✅ LLM ready")
    return llm

# LAZY LOAD FAISS
def get_faiss():
    """Lazy import FAISS"""
    from langchain_community.vectorstores import FAISS
    return FAISS

# Flask app setup
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global state
current_resume_text = ""
original_resume_text = ""  # Keep original unchanged
current_suggestions = []
vectorstore = None
original_pdf_path = ""
highlighted_suggestions = {}
pending_edits = []  # Queue of edits to apply on export


# ============================================================================
# PDF EDITING - BATCH APPROACH
# ============================================================================

def apply_batch_edits_to_pdf(input_pdf_path, output_pdf_path, edits_list):
    """
    Apply edits to PDF using redaction method for font preservation
    """
    try:
        fitz.TOOLS.set_small_glyph_heights(True)
        
        doc = fitz.open(input_pdf_path)
        total_replaced = 0
        
        for edit in edits_list:
            original_text = edit['original']
            new_text = edit['new']
            text_replaced = False
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Search for the original text
                text_instances = page.search_for(original_text)
                
                if text_instances:
                    text_replaced = True
                    
                    for inst in text_instances:
                        rect = fitz.Rect(inst)
                        
                        # Extract font properties
                        try:
                            blocks = page.get_text("dict", clip=rect)["blocks"]
                            
                            all_fonts = []
                            all_sizes = []
                            all_colors = []
                            
                            for block in blocks:
                                if block.get("type") == 0:  # text block
                                    for line in block.get("lines", []):
                                        for span in line.get("spans", []):
                                            font = span.get("font", "Helvetica")
                                            size = span.get("size", 0)
                                            color = span.get("color", 0)
                                            
                                            if size > 0:
                                                all_fonts.append(font)
                                                all_sizes.append(size)
                                                all_colors.append(color)
                            
                            # Use maximum size (most prominent)
                            if all_sizes:
                                font_size = max(all_sizes)
                                print(f"📊 Font Analysis: sizes={all_sizes}, using {font_size:.1f}")
                            else:
                                # Estimate from rectangle height
                                font_size = rect.height * 0.8
                                print(f"⚠️  Estimating font: {font_size:.1f} from height {rect.height:.1f}")
                            
                            # Get text color
                            if all_colors:
                                color_int = all_colors[0]
                                r = ((color_int >> 16) & 0xFF) / 255.0
                                g = ((color_int >> 8) & 0xFF) / 255.0
                                b = (color_int & 0xFF) / 255.0
                                text_color = (r, g, b)
                            else:
                                text_color = (0, 0, 0)
                            
                        except Exception as e:
                            print(f"❌ Font extraction failed: {e}, using fallback")
                            font_size = rect.height * 0.8
                            text_color = (0, 0, 0)
                        
                        # Ensure reasonable bounds
                        if font_size < 7:
                            font_size = 9
                        elif font_size > 22:
                            font_size = 14
                        
                        # Calculate text expansion
                        len_ratio = len(new_text) / max(len(original_text), 1)
                        original_font_size = font_size
                        
                        # Smart expansion with minimal font change
                        expanded_rect = rect
                        if len_ratio > 1.1:
                            extra_width = rect.width * (len_ratio - 1) * 0.6
                            page_width = page.rect.width
                            
                            if rect.x1 + extra_width < page_width - 20:
                                expanded_rect = fitz.Rect(
                                    rect.x0,
                                    rect.y0,
                                    rect.x1 + extra_width,
                                    rect.y1
                                )
                                print(f"   ✓ Expanded rect by {extra_width:.1f}pt, keeping font at {font_size:.1f}")
                            else:
                                # Gentle font reduction
                                reduction = min(0.95, 1.0 / (len_ratio ** 0.3))
                                font_size = font_size * reduction
                                print(f"   ⚠️  Limited space, reducing font to {font_size:.1f}")
                        
                        font_size = round(font_size, 1)
                        
                        print(f"\n{'='*70}")
                        print(f"📝 REPLACEMENT:")
                        print(f"   Original: '{original_text}'")
                        print(f"   New:      '{new_text}'")
                        print(f"   Length:   {len(original_text)} → {len(new_text)} (ratio: {len_ratio:.2f})")
                        print(f"   Font:     {original_font_size:.1f} → {font_size:.1f}")
                        print(f"{'='*70}\n")
                        
                        # Apply redaction
                        page.add_redact_annot(
                            expanded_rect,
                            text=new_text,
                            fontname="helv",
                            fontsize=font_size,
                            text_color=text_color,
                            align=fitz.TEXT_ALIGN_LEFT,
                            fill=(1, 1, 1)
                        )
                        
                        total_replaced += 1
                        break
                
                if text_replaced:
                    break
        
        # Apply all redactions
        print("\n🔄 Applying all redactions...")
        for page in doc:
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE
            )
        
        # Save with compression
        print("💾 Saving PDF...")
        doc.save(
            output_pdf_path,
            garbage=4,
            deflate=True,
            clean=True,
            pretty=False
        )
        doc.close()
        
        print(f"\n✅ Successfully applied {total_replaced} edits to PDF\n")
        return True, total_replaced
        
    except Exception as e:
        print(f"\n❌ PDF batch editing error: {e}")
        traceback.print_exc()
        return False, 0


def create_pdf_from_text(text, output_path):
    """Fallback: Create PDF from text"""
    try:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        elements = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=14,
            spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            leading=20,
            spaceAfter=12,
            textColor='#1a5490',
        ))
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 0.2*inch))
                continue
            
            if line.isupper() or line.endswith(':'):
                p = Paragraph(line, styles['CustomHeading'])
            else:
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                p = Paragraph(line, styles['CustomBody'])
            
            elements.append(p)
        
        doc.build(elements)
        return True
    except Exception as e:
        print(f"PDF creation error: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# AGENT TOOLS
# ============================================================================

class ResumeAnalyzerTools:
    """Tools for resume analysis"""
    
    def __init__(self, resume_text: str):
        self.resume_text = resume_text
        self.analysis_history = []
    
    def analyze_grammar_realtime(self, section: str = "all") -> str:
        """Real-time grammar analysis"""
        prompt = f"""
Analyze this resume for grammar, spelling, and punctuation errors.

Resume text:
{self.resume_text[:3000]}

CRITICAL RULES:
1. Extract the EXACT text as it appears (match punctuation exactly)
2. Keep suggestions SAME LENGTH as original (±20% max)
3. Do NOT add new information or expand content
4. Only fix the specific error, nothing more

Find up to 5 clear errors. For each:
- original: EXACT text with error (3-20 words, must match resume exactly)
- suggested: corrected version (SAME approximate length)
- reason: brief explanation (one sentence)
- type: "Grammar"

Return ONLY this JSON format:
[
    {{"original": "exact error text", "suggested": "corrected text", "reason": "explanation", "type": "Grammar"}}
]

If no errors found, return: []
"""
        try:
            result = get_llm().invoke(prompt).content.strip()
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            test_parse = json.loads(result)
            self.analysis_history.append({"tool": "analyze_grammar_realtime", "result": result})
            return result
        except Exception as e:
            print(f"Grammar analysis error: {e}")
            return "[]"
    
    def analyze_language_quality(self, focus: str = "professional") -> str:
        """Language quality analysis"""
        prompt = f"""
Analyze language quality in this resume. Find weak verbs and passive voice.

Resume text:
{self.resume_text[:3000]}

CRITICAL RULES:
1. Extract EXACT text as it appears
2. Keep suggestions SAME LENGTH as original (±20% max)
3. Only replace weak words, don't add new information
4. Maintain the original sentence structure

Find up to 5 improvements. For each:
- original: EXACT weak text (3-20 words, must match exactly)
- suggested: stronger version (SAME approximate length)
- reason: why it's better (one sentence)
- type: "Language"

Return ONLY this JSON:
[
    {{"original": "weak text", "suggested": "improved text", "reason": "explanation", "type": "Language"}}
]

If nothing to improve, return: []
"""
        try:
            result = get_llm().invoke(prompt).content.strip()
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            test_parse = json.loads(result)
            self.analysis_history.append({"tool": "analyze_language_quality", "result": result})
            return result
        except Exception as e:
            print(f"Language analysis error: {e}")
            return "[]"
    
    def analyze_achievements(self) -> str:
        """Achievement analysis"""
        prompt = f"""
Find achievements in this resume that need metrics or quantification.

Resume text:
{self.resume_text[:3000]}

CRITICAL RULES:
1. Extract EXACT text as it appears
2. Keep suggestions SAME LENGTH as original (±30% max for adding metrics)
3. Only add specific numbers/percentages, don't expand the sentence
4. Maintain original structure

Find up to 5 achievements to improve. For each:
- original: EXACT achievement text (3-20 words, must match exactly)
- suggested: version with concise metrics (similar length)
- reason: why it's better (one sentence)
- type: "Achievement"

Return ONLY this JSON:
[
    {{"original": "achievement text", "suggested": "with metrics", "reason": "explanation", "type": "Achievement"}}
]

If nothing to improve, return: []
"""
        try:
            result = get_llm().invoke(prompt).content.strip()
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            test_parse = json.loads(result)
            self.analysis_history.append({"tool": "analyze_achievements", "result": result})
            return result
        except Exception as e:
            print(f"Achievement analysis error: {e}")
            return "[]"


# ============================================================================
# AGENT WORKFLOW
# ============================================================================

def agent_analyze_resume(user_query: str, resume_text: str) -> Dict[str, Any]:
    """Main agent workflow"""
    
    try:
        tools_instance = ResumeAnalyzerTools(resume_text)
        
        query_lower = user_query.lower()
        suggestions = []
        
        if 'grammar' in query_lower or 'fix' in query_lower or 'error' in query_lower or 'spelling' in query_lower:
            print("Running grammar analysis...")
            result = tools_instance.analyze_grammar_realtime()
            try:
                grammar_suggestions = json.loads(result)
                suggestions.extend(grammar_suggestions)
            except:
                print(f"Grammar parse error: {result}")
        
        if 'language' in query_lower or 'improve' in query_lower or 'enhance' in query_lower or 'better' in query_lower:
            print("Running language analysis...")
            result = tools_instance.analyze_language_quality()
            try:
                language_suggestions = json.loads(result)
                suggestions.extend(language_suggestions)
            except:
                print(f"Language parse error: {result}")
        
        if 'achievement' in query_lower or 'metric' in query_lower or 'quantif' in query_lower or 'number' in query_lower:
            print("Running achievement analysis...")
            result = tools_instance.analyze_achievements()
            try:
                achievement_suggestions = json.loads(result)
                suggestions.extend(achievement_suggestions)
            except:
                print(f"Achievement parse error: {result}")
        
        if not suggestions:
            print("Running default grammar analysis...")
            result = tools_instance.analyze_grammar_realtime()
            try:
                suggestions = json.loads(result)
            except:
                suggestions = []
        
        for idx, sug in enumerate(suggestions):
            sug['id'] = f"sug_{idx}_{hash(sug.get('original', ''))}"
        
        # Validate suggestions
        validated = []
        for sug in suggestions:
            original = sug.get('original', '').strip()
            if original and original in resume_text:
                suggested = sug.get('suggested', '').strip()
                len_diff = abs(len(suggested) - len(original))
                if len_diff < len(original) * 0.5:
                    validated.append(sug)
                else:
                    print(f"Skipping suggestion with large length difference: {len_diff}")
            else:
                print(f"Skipping invalid suggestion: {original[:50]}")
        
        response_text = f"I found {len(validated)} improvement{'s' if len(validated) != 1 else ''} in your resume!"
        if len(validated) == 0:
            response_text = "Your resume looks good! I couldn't find any major issues to fix."
        
        return {
            "response": response_text,
            "suggestions": validated
        }
        
    except Exception as e:
        print(f"Agent error: {e}")
        traceback.print_exc()
        return {
            "response": "I analyzed your resume but encountered an issue. Please try a more specific request.",
            "suggestions": []
        }


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/health')
def health():
    """Simple health check for Render"""
    return jsonify({'status': 'ok'}), 200


@app.route('/warmup')
def warmup():
    """Warm up the app by loading all models"""
    try:
        # Load embeddings
        get_embeddings()
        
        # Load LLM and test it
        test_llm = get_llm()
        test_llm.invoke("test")
        
        return jsonify({
            'status': 'warmed up',
            'message': 'All models loaded successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_resume_text, original_resume_text, vectorstore, original_pdf_path, highlighted_suggestions, pending_edits
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            original_pdf_path = file_path
            
            resume_text = extract_text_from_pdf(file_path)
            current_resume_text = resume_text
            original_resume_text = resume_text  # Keep original
            
            splitted_text = text_splitter.split_text(resume_text)
            FAISS = get_faiss()
            vectorstore = FAISS.from_texts(splitted_text, get_embeddings())
            # DON'T save to local - keep in memory only
            
            highlighted_suggestions = {}
            pending_edits = []  # Clear pending edits
            
            return jsonify({
                'success': True,
                'resume_text': resume_text,
                'filename': filename
            })
        except Exception as e:
            print(f"Upload error: {e}")
            return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Please upload a PDF file'}), 400


@app.route('/ask', methods=['POST'])
def ask_query():
    global current_resume_text, current_suggestions, highlighted_suggestions
    
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    if not current_resume_text:
        return jsonify({'error': 'Please upload a resume first'}), 400
    
    try:
        print(f"Processing query: {query}")
        
        improvement_keywords = ['fix', 'improve', 'change', 'enhance', 'better', 'correct', 
                               'grammar', 'mistake', 'rewrite', 'suggest', 'upgrade', 
                               'error', 'spelling', 'language', 'weak', 'metric', 'quantif',
                               'achievement', 'format', 'structure']
        
        query_lower = query.lower()
        is_improvement_request = any(keyword in query_lower for keyword in improvement_keywords)
        
        question_indicators = ['what', 'who', 'where', 'when', 'why', 'how', 'describe', 
                              'explain', 'tell me', 'show me', 'summarize', 'summary',
                              'list', 'find', 'search', 'which', 'does', 'is', 'are',
                              'can you', 'could you', 'would you']
        is_question = any(indicator in query_lower for indicator in question_indicators)
        
        if is_question and not is_improvement_request:
            print("Processing as Q&A query...")
            try:
                if vectorstore is None:
                    raise Exception("Vector store not initialized")
                
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                qa_prompt = f"""Based on the following resume content, answer this question:

Question: {query}

Resume excerpt:
{context}

Provide a clear, concise answer based only on the information in the resume."""
                
                answer = get_llm().invoke(qa_prompt).content
                
                return jsonify({
                    'response': answer,
                    'suggestions': [],
                    'is_qa': True
                })
                
            except Exception as e:
                print(f"RAG error: {e}")
                fallback_prompt = f"""Answer this question about the resume:

Question: {query}

Resume:
{current_resume_text[:2000]}

Provide a helpful answer based on the resume content."""
                
                answer = get_llm().invoke(fallback_prompt).content
                return jsonify({
                    'response': answer,
                    'suggestions': [],
                    'is_qa': True
                })
        
        else:
            print("Processing as improvement request...")
            result = agent_analyze_resume(query, current_resume_text)
            suggestions = result.get('suggestions', [])
            
            print(f"Found {len(suggestions)} suggestions")
            
            highlighted_suggestions = {}
            for sug in suggestions:
                if 'id' in sug:
                    highlighted_suggestions[sug['id']] = sug
            
            current_suggestions = suggestions
            
            return jsonify({
                'response': result.get('response', 'Analysis complete!'),
                'suggestions': suggestions,
                'highlighted_text': current_resume_text,
                'is_qa': False
            })
        
    except Exception as e:
        print(f"Query error: {e}")
        traceback.print_exc()
        return jsonify({
            'response': f'I encountered an error while processing your request.',
            'suggestions': [],
            'error': str(e)
        }), 500


@app.route('/apply_suggestion', methods=['POST'])
def apply_suggestion():
    """Accept a suggestion and add it to pending edits queue"""
    global current_resume_text, highlighted_suggestions, vectorstore, pending_edits
    
    data = request.get_json()
    suggestion_id = data.get('suggestion_id')
    
    if not suggestion_id or suggestion_id not in highlighted_suggestions:
        return jsonify({'error': 'Invalid suggestion'}), 400
    
    suggestion = highlighted_suggestions[suggestion_id]
    original = suggestion['original']
    suggested = suggestion['suggested']
    
    print(f"\n{'='*60}")
    print(f"Accepting suggestion (queued for export):")
    print(f"Original: '{original}'")
    print(f"Suggested: '{suggested}'")
    print(f"{'='*60}\n")
    
    # Verify text exists in CURRENT resume text
    if original not in current_resume_text:
        return jsonify({'error': 'Text not found in current resume version'}), 404
    
    # Update the in-memory text immediately for preview
    new_text = current_resume_text.replace(original, suggested, 1)
    current_resume_text = new_text
    
    # Add to pending edits queue
    pending_edits.append({
        'original': original,
        'new': suggested,
        'type': suggestion.get('type', 'Edit')
    })
    
    # Update vector store with new text (in memory only)
    splitted_text = text_splitter.split_text(new_text)
    FAISS = get_faiss()
    vectorstore = FAISS.from_texts(splitted_text, get_embeddings())
    
    # Remove from highlighted suggestions
    del highlighted_suggestions[suggestion_id]
    
    print(f"✅ Edit #{len(pending_edits)} queued for export")
    
    return jsonify({
        'success': True,
        'updated_text': new_text,
        'pending_count': len(pending_edits),
        'message': f'Edit accepted! {len(pending_edits)} change(s) will be applied when you export.'
    })


@app.route('/get_pdf', methods=['GET'])
def get_pdf():
    """Return original PDF (changes only applied on export)"""
    global original_pdf_path
    if original_pdf_path and os.path.exists(original_pdf_path):
        return send_file(original_pdf_path, mimetype='application/pdf')
    return jsonify({'error': 'No PDF available'}), 404


@app.route('/export', methods=['POST'])
def export_resume():
    """Apply ALL pending edits in batch and export"""
    global original_pdf_path, pending_edits
    
    if not original_pdf_path or not os.path.exists(original_pdf_path):
        return jsonify({'error': 'No resume to export'}), 400
    
    try:
        if len(pending_edits) == 0:
            # No edits, return original
            return send_file(
                original_pdf_path, 
                as_attachment=True, 
                download_name='resume.pdf',
                mimetype='application/pdf'
            )
        
        # Create output path
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'improved_resume.pdf')
        
        print("\n" + "="*80)
        print(f"🚀 EXPORTING WITH {len(pending_edits)} EDITS")
        print("="*80)
        
        # Try redaction method (most accurate)
        print("\n📝 Applying edits...")
        success, count = apply_batch_edits_to_pdf(original_pdf_path, output_path, pending_edits)
        
        if not success:
            # Fallback: Create from text
            print("\n⚠️  PDF edit failed, creating from text...")
            create_pdf_from_text(current_resume_text, output_path)
            success = True
            count = len(pending_edits)
        
        if success:
            print(f"\n✅ Export successful! Applied {count} edits")
            print("="*80 + "\n")
            
            return send_file(
                output_path, 
                as_attachment=True, 
                download_name='improved_resume.pdf',
                mimetype='application/pdf'
            )
        else:
            raise Exception("All export methods failed")
            
    except Exception as e:
        print(f"\n❌ Export error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset_resume():
    """Reset to original state"""
    global current_resume_text, original_resume_text, pending_edits, vectorstore, highlighted_suggestions
    
    if not original_resume_text:
        return jsonify({'error': 'No original resume found'}), 404
    
    try:
        current_resume_text = original_resume_text
        pending_edits = []
        highlighted_suggestions = {}
        
        # Update vector store
        splitted_text = text_splitter.split_text(original_resume_text)
        FAISS = get_faiss()
        vectorstore = FAISS.from_texts(splitted_text, get_embeddings())
        
        return jsonify({
            'success': True,
            'message': 'Resume reset to original version',
            'resume_text': original_resume_text
        })
    except Exception as e:
        print(f"Reset error: {e}")
        return jsonify({'error': 'Failed to reset resume'}), 500


@app.route('/pending_edits', methods=['GET'])
def get_pending_edits():
    """Get count of pending edits"""
    return jsonify({
        'count': len(pending_edits),
        'edits': pending_edits
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}...")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, _quiet=True)
