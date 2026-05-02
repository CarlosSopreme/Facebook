from google.cloud import bigquery
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
import os
from datetime import datetime, timezone, timedelta
from anthropic import Anthropic
import dspy
from pydantic import BaseModel
import json
from typing import List, Dict, Optional
import re

# Database imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class FBPost(Base):
    __tablename__ = "fb_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    ad_account_name = Column(String, index=True)
    campaign_name = Column(String, index=True)
    ad_set_name = Column(String)
    ad_name = Column(String)
    page_id = Column(String, index=True)
    post_id = Column(String, unique=True, index=True)
    object_story_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResponseEntry(Base):
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(Text)
    action = Column(String, index=True)
    reply = Column(Text)
    reasoning = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# NOW import marketing router after database is set up
try:
    from marketing_analytics import marketing_router
    MARKETING_AVAILABLE = True
    print("✅ Marketing analytics loaded successfully")
except Exception as e:
    MARKETING_AVAILABLE = False
    marketing_router = None
    print(f"⚠️ Marketing analytics failed to load: {e}")

# Centralized Business Rules - Single Source of Truth
BUSINESS_RULES = """
COMPANY POSITION:
- LeaseEnd helps drivers get loans in their name with competitive options, completely online
- We connect customers with lenders (NOT third-party financing)
- Challenge false information politely

ACTIONS:
- REPLY: Genuine questions, prospects, positive feedback, correctable misinformation
- REACT: Positive comments needing no response
- DELETE: Accusations, spam, hostility, excessive arguing, brief negatives, negative Leaseend experiences, chidlish comments
- LEAVE_ALONE: Harmless off-topic or neutral tagged comments, so anything with a name in front or referencing another name

BRAND VOICE:
- Professional but conversational
- Confident in lease analysis expertise
- Concise responses focused on THEIR specific situation
- Loan facilitators, not lenders

RESPONSE GUIDELINES (if REPLY):
- Address their specific concern or statement directly
- Focus on providing genuine insight that positions us as an authority
- Don't need to mention what Lease End does unless specifically asked
- NEVER include dollar amounts, percentages, or rates
- If alternative options besides Lease End are mentioned delete the comment
- If asked about costs or fees of our service respond with this default answer, "We charge a doc fee, which pays for the service we provide. The only additional amount you will pay beyond the price of your vehicle, would be for tax, title and registration fees, which you would pay regardless of whatever route you choose."
- 3 sentences maximum

COMPETITIVE ADVANTAGES:
- vs Dealerships: No pressure, transparent, 100% online
- vs Credit Unions: No membership required, flexible, fast
- vs Banks: Competitive rates, simple, customer-focused

WEBSITE CTA "fill out form on our site" - ONLY for:
- High-intent prospects showing clear purchase intent
- NOT for casual browsers or general questions

PHONE NUMBER (844) 679-1188 - ONLY for:
- Explicit contact requests ("call me", "speak to someone")
- Hesitation ("not sure", "worried", "what's the catch")
- Confusion ("don't understand", "complicated")
- Urgency ("urgent", "asap")
- NOT for general interest ("interested", "how much", "can I qualify")
"""

# Custom Anthropic LM for DSPy (unchanged)
class CustomAnthropic(dspy.LM):
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.kwargs = {"max_tokens": 1000}
        self.history = []
        
    def basic_request(self, prompt, **kwargs):
        """Core method that handles the actual API call"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return "Error generating response"
    
    def __call__(self, prompt=None, messages=None, **kwargs):
        """Handle different calling patterns DSPy might use"""
        if prompt is None and messages is not None:
            if isinstance(messages, list) and len(messages) > 0:
                prompt = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
            else:
                prompt = str(messages)
        elif prompt is None:
            prompt = ""
            
        result = self.basic_request(prompt, **kwargs)
        return [result]
    
    def generate(self, prompt, **kwargs):
        return self.__call__(prompt, **kwargs)
    
    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

# Configure DSPy
try:
    claude = CustomAnthropic(api_key=ANTHROPIC_API_KEY)
    dspy.settings.configure(lm=claude)
except Exception as e:
    raise ValueError(f"Failed to configure DSPy: {str(e)}")

# Load training data from database
def load_training_data():
    """Load training examples from database"""
    print("🔄 Loading training data from database...")
    
    try:
        db = SessionLocal()
        responses = db.query(ResponseEntry).all()
        db.close()
        
        training_data = []
        for response in responses:
            training_data.append({
                'comment': response.comment,
                'action': response.action,
                'reply': response.reply,
                'reasoning': response.reasoning
            })
        
        print(f"✅ Loaded {len(training_data)} training examples from database")
        return training_data
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return []

# Global training data
TRAINING_DATA = load_training_data()

def reload_training_data():
    """Reload training data from database"""
    global TRAINING_DATA
    TRAINING_DATA = load_training_data()
    return len(TRAINING_DATA)

def filter_numerical_values(text):
    """Remove any numerical values (dollars, percentages, rates) from response text"""
    
    # Remove dollar amounts ($X, $X.XX, $X,XXX, etc.)
    text = re.sub(r'\$[\d,]+(?:\.\d{2})?', '', text)
    
    # Remove percentages (X%, X.X%, XX.XX%, etc.)
    text = re.sub(r'\b\d+(?:\.\d+)?%', '', text)
    
    # Remove APR/interest rate patterns (X.X% APR, X% interest, etc.)
    text = re.sub(r'\b\d+(?:\.\d+)?\s*%?\s*(?:APR|apr|interest|rate)', '', text)
    
    # Remove standalone numbers that might be rates (like "4.5" or "6.2")
    text = re.sub(r'\b\d+\.\d+\b(?=\s*(?:rate|APR|interest|%|percent))', '', text)
    
    # Clean up any double spaces or awkward spacing created by removals
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    
    return text.strip()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def process_comment_with_ai(comment, commentId=""):
    """Unified comment analysis and response generation using centralized rules"""
    
    prompt = f"""You are analyzing and responding to comments for LeaseEnd.com, which helps drivers get loans for lease buyouts.

{BUSINESS_RULES}

COMMENT: "{comment}"

Respond in this JSON format: 
{{
    "sentiment": "...", 
    "action": "REPLY/REACT/DELETE/LEAVE_ALONE", 
    "reasoning": "...", 
    "high_intent": true/false, 
    "needs_phone": true/false,
    "reply": "..." (only include if action is REPLY, otherwise empty string)
}}"""

    try:
        response = claude.basic_request(prompt)
        response_clean = response.strip()
        
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
        
        try:
            result = json.loads(response_clean)
            
            # Clean the reply if it exists
            reply = result.get('reply', '')
            if reply:
                reply = filter_numerical_values(reply.strip())
                # Ensure phone number format is correct if present
                if '844' in reply and '679-1188' in reply:
                    reply = re.sub(r'\(?844\)?[-.\s]*679[-.\s]*1188', '(844) 679-1188', reply)
            
            return {
                'sentiment': result.get('sentiment', 'Neutral'),
                'action': result.get('action', 'LEAVE_ALONE'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'high_intent': result.get('high_intent', False),
                'needs_phone': result.get('needs_phone', False),
                'reply': reply
            }
            
        except json.JSONDecodeError:
            # Enhanced fallback parsing
            if any(word in response.upper() for word in ['DELETE', 'SCAM', 'FALSE INFO', 'LIES', 'FRAUD']):
                action = 'DELETE'
                reasoning = "Detected negative/accusatory content"
            elif 'REPLY' in response.upper():
                action = 'REPLY'
                reasoning = "Detected genuine prospect or question"
            else:
                action = 'LEAVE_ALONE'
                reasoning = "Fallback classification"
                
            return {
                'sentiment': 'Neutral',
                'action': action,
                'reasoning': reasoning,
                'high_intent': False,
                'needs_phone': False,
                'reply': "Thank you for your comment! We'd be happy to help analyze your specific lease situation." if action == 'REPLY' else ""
            }
            
    except Exception as e:
        return {
            'sentiment': 'Neutral',
            'action': 'LEAVE_ALONE',
            'reasoning': f'Processing error: {str(e)}',
            'high_intent': False,
            'needs_phone': False,
            'reply': ""
        }

# Pydantic Models
class CommentRequest(BaseModel):
    comment: Optional[str] = None
    message: Optional[str] = None
    commentId: Optional[str] = None
    postId: Optional[str] = None
    created_time: Optional[str] = ""
    memory_context: Optional[str] = ""
    
    def get_comment_text(self) -> str:
        return (self.comment or self.message or "No message content").strip()
    
    def get_comment_id(self) -> str:
        return (self.commentId or self.postId or "unknown").strip()

class ProcessedComment(BaseModel):
    commentId: str
    original_comment: str
    category: str
    action: str
    reply: str
    confidence_score: float
    approved: str
    reasoning: str

class FeedbackRequest(BaseModel):
    original_comment: str
    original_response: str = ""
    original_action: str
    feedback_text: str
    commentId: str
    current_version: str = "v1"
    
    class Config:
        extra = "ignore"

class ApproveRequest(BaseModel):
    original_comment: str
    action: str
    reply: str
    reasoning: str = ""
    created_time: Optional[str] = ""
    
    class Config:
        extra = "ignore"

class FBPostCreate(BaseModel):
    ad_account_name: str
    campaign_name: str
    ad_set_name: str
    ad_name: str
    page_id: str
    post_id: str
    object_story_id: str

class ResponseCreate(BaseModel):
    comment: str
    action: str
    reply: str
    reasoning: Optional[str] = ""

app = FastAPI()

# Include marketing router only if it loaded successfully
if MARKETING_AVAILABLE and marketing_router:
    app.include_router(marketing_router)

@app.get("/")
def read_root():
    return {
        "message": "Lease End AI Assistant - CENTRALIZED RULES + Marketing Analytics",
        "version": "32.0-WITH-MARKETING-ANALYTICS", 
        "training_examples": len(TRAINING_DATA),
        "status": "RUNNING",
        "features": [
            "Centralized Business Rules", 
            "Consistent Endpoints", 
            "Simplified Architecture",
            "Marketing Trends Analysis",  # NEW
            "BigQuery Integration"        # NEW
        ],
        "endpoints": {
            "ai_assistant": [
                "/process-comment",
                "/process-feedback", 
                "/approve-response"
            ],
            "marketing_analytics": [    # NEW SECTION
                "/marketing/analyze-trends",
                "/marketing/trend-history",
                "/marketing/test-bigquery"
            ],
            "data_management": [
                "/fb-posts",
                "/responses",
                "/stats"
            ]
        }
    }

@app.get("/ping")
@app.post("/ping") 
@app.head("/ping")
def ping():
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "App is running"
    }

@app.get("/health")
def health_check():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "training_examples": len(TRAINING_DATA)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "error",
            "error": str(e)
        }

@app.get("/fb-posts")
async def get_fb_posts():
    """Get all FB posts from database"""
    db = SessionLocal()
    try:
        posts = db.query(FBPost).all()
        
        return {
            "success": True,
            "count": len(posts),
            "data": [
                {
                    "Ad account name": post.ad_account_name,
                    "Campaign name": post.campaign_name,
                    "Ad set name": post.ad_set_name,
                    "Ad name": post.ad_name,
                    "Page ID": post.page_id,
                    "Post Id": post.post_id,
                    "Object Story ID": post.object_story_id
                }
                for post in posts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/fb-posts/add")
async def add_fb_post(post: FBPostCreate):
    """Add new FB post to database"""
    try:
        db = SessionLocal()
        
        db_post = FBPost(
            ad_account_name=post.ad_account_name,
            campaign_name=post.campaign_name,
            ad_set_name=post.ad_set_name,
            ad_name=post.ad_name,
            page_id=post.page_id,
            post_id=post.post_id,
            object_story_id=post.object_story_id
        )
        
        db.add(db_post)
        db.commit()
        db.close()
        
        return {
            "success": True,
            "message": "FB post added successfully",
            "post_id": post.post_id,
            "status": "new"
        }
        
    except IntegrityError:
        db.rollback()
        db.close()
        return {
            "success": True,
            "message": f"Post {post.post_id} already exists (duplicate skipped)",
            "post_id": post.post_id,
            "status": "duplicate_skipped"
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/fb-posts/clear-recent")
async def clear_recent_posts():
    """Delete FB posts created in the past 24 hours"""
    try:
        db = SessionLocal()
        
        # Calculate 24 hours ago
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        
        # Find posts created in the past 24 hours
        recent_posts = db.query(FBPost).filter(FBPost.created_at >= twenty_four_hours_ago).all()
        deleted_count = len(recent_posts)
        
        # Delete them
        db.query(FBPost).filter(FBPost.created_at >= twenty_four_hours_ago).delete()
        db.commit()
        db.close()
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} posts created in the past 24 hours",
            "deleted_count": deleted_count,
            "cutoff_time": twenty_four_hours_ago.isoformat()
        }
        
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/responses")
async def get_responses():
    """Get all response training data"""
    db = SessionLocal()
    try:
        responses = db.query(ResponseEntry).all()
        
        return {
            "success": True,
            "count": len(responses),
            "data": [
                {
                    "comment": resp.comment,
                    "action": resp.action,
                    "reply": resp.reply
                }
                for resp in responses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/responses/add")
async def add_response(response: ResponseCreate):
    """Add new response training data"""
    db = SessionLocal()
    try:
        db_response = ResponseEntry(
            comment=response.comment,
            action=response.action,
            reply=response.reply,
            reasoning=response.reasoning
        )
        
        db.add(db_response)
        db.commit()
        
        new_count = reload_training_data()
        
        return {
            "success": True,
            "message": "Response added successfully",
            "new_training_count": new_count
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.post("/reload-training-data")
async def reload_training_data_endpoint():
    """Manually reload training data from database"""
    try:
        new_count = reload_training_data()
        return {
            "success": True,
            "message": "Training data reloaded from database",
            "total_examples": new_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading training data: {str(e)}")

@app.post("/process-comment", response_model=ProcessedComment)
async def process_comment(request: CommentRequest):
    """Enhanced comment processing using centralized business rules"""
    try:
        comment_text = request.get_comment_text()
        comment_id = request.get_comment_id()
        
        # Use centralized function
        ai_result = process_comment_with_ai(comment_text, comment_id)
        
        sentiment = ai_result['sentiment']
        action = ai_result['action'].lower()
        reasoning = ai_result['reasoning']
        reply_text = ai_result['reply']  # Already generated and cleaned
        
        # Map actions to our system
        action_mapping = {
            'reply': 'respond',
            'react': 'react', 
            'delete': 'delete',
            'leave_alone': 'leave_alone'
        }
        
        mapped_action = action_mapping.get(action, 'leave_alone')
        confidence_score = 0.9 if mapped_action == 'respond' else 0.85
        
        return ProcessedComment(
            commentId=comment_id,
            original_comment=comment_text,
            category=sentiment.lower(),
            action=mapped_action,
            reply=reply_text,
            confidence_score=confidence_score,
            approved="pending",
            reasoning=reasoning
        )
        
    except Exception as e:
        return ProcessedComment(
            commentId=request.get_comment_id(),
            original_comment=request.get_comment_text(),
            category="error",
            action="leave_alone",
            reply="We can help analyze your specific lease situation.",
            confidence_score=0.0,
            approved="pending",
            reasoning=f"Error: {str(e)}"
        )

@app.post("/process-feedback")
async def process_feedback(request: FeedbackRequest):
    """Enhanced feedback processing using centralized business rules"""
    try:
        original_comment = request.original_comment.strip()
        original_response = request.original_response.strip() if request.original_response else ""
        original_action = request.original_action.strip().lower()
        feedback_text = request.feedback_text.strip()
        
        feedback_prompt = f"""You are improving a response based on human feedback for LeaseEnd.com.

ORIGINAL COMMENT: "{original_comment}"
YOUR ORIGINAL RESPONSE: "{original_response}"
YOUR ORIGINAL ACTION: "{original_action}"
HUMAN FEEDBACK: "{feedback_text}"

{BUSINESS_RULES}

Generate an IMPROVED response incorporating the feedback while following the guidelines above. Answer the original comment NOT the feedback.

Respond in JSON: {{"sentiment": "...", "action": "REPLY/REACT/DELETE/LEAVE_ALONE", "reply": "...", "reasoning": "...", "confidence": 0.85, "needs_phone": true/false}}"""

        improved_response = claude.basic_request(feedback_prompt)
        
        try:
            response_clean = improved_response.strip()
            if response_clean.startswith('```'):
                lines = response_clean.split('\n')
                response_clean = '\n'.join([line for line in lines if not line.startswith('```')])
            
            if not response_clean.startswith('{'):
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    response_clean = response_clean[json_start:json_end]
            
            result = json.loads(response_clean)
            
            action_mapping = {
                'reply': 'respond',
                'react': 'react', 
                'delete': 'delete',
                'leave_alone': 'leave_alone'
            }
            
            improved_action = action_mapping.get(result.get('action', 'leave_alone').lower(), 'leave_alone')
            improved_reply = filter_numerical_values(result.get('reply', ''))
            
            # Fix phone number format if present
            if '844' in improved_reply and '679-1188' in improved_reply:
                improved_reply = re.sub(r'\(?844\)?[-.\s]*679[-.\s]*1188', '(844) 679-1188', improved_reply)
            
            current_version_num = int(request.current_version.replace('v', '')) if request.current_version.startswith('v') else 1
            new_version = f"v{current_version_num + 1}"
            
            return {
                "commentId": request.commentId,
                "original_comment": original_comment,
                "category": result.get('sentiment', 'neutral').lower(),
                "action": improved_action,
                "reply": improved_reply,
                "confidence_score": float(result.get('confidence', 0.85)),
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": new_version,
                "reasoning": result.get('reasoning', 'Applied feedback with centralized business rules'),
                "feedback_processed": True,
                "success": True
            }
            
        except json.JSONDecodeError as e:
            new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
            return {
                "commentId": request.commentId,
                "original_comment": original_comment,
                "category": "neutral",
                "action": "leave_alone",
                "reply": "We can help analyze your specific lease situation. Call (844) 679-1188 if you have questions.",
                "confidence_score": 0.5,
                "approved": "pending",
                "feedback_text": feedback_text,
                "version": f"v{new_version_num}",
                "reasoning": "Fallback response with proper phone format",
                "success": False
            }
            
    except Exception as e:
        new_version_num = int(request.current_version.replace('v', '')) + 1 if request.current_version.startswith('v') else 2
        return {
            "commentId": request.commentId,
            "original_comment": request.original_comment,
            "category": "error", 
            "action": "leave_alone",
            "reply": "We can help analyze your specific lease situation.",
            "confidence_score": 0.0,
            "approved": "pending",
            "feedback_text": request.feedback_text,
            "version": f"v{new_version_num}",
            "reasoning": "Error in feedback processing",
            "error": str(e),
            "success": False
        }

@app.post("/approve-response")
async def approve_response(request: ApproveRequest):
    """Approve and store a response for training"""
    global TRAINING_DATA
    
    db = SessionLocal()
    try:
        # Check if comment already exists
        existing_comment = db.query(ResponseEntry).filter(
            ResponseEntry.comment == request.original_comment
        ).first()
        
        if existing_comment:
            return {
                "status": "duplicate_skipped",
                "message": f"Comment already exists in training data (ID: {existing_comment.id})",
                "training_examples": len(TRAINING_DATA),
                "existing_action": existing_comment.action,
                "duplicate": True
            }
        
        comment_created_at = datetime.utcnow()
        if request.created_time:
            try:
                comment_created_at = datetime.fromisoformat(request.created_time.replace('Z', '+00:00'))
            except:
                pass
        
        response_entry = ResponseEntry(
            comment=request.original_comment,
            action=request.action,
            reply=request.reply,
            reasoning=request.reasoning,
            created_at=comment_created_at
        )
        db.add(response_entry)
        db.commit()
        
        TRAINING_DATA = load_training_data()
        
        return {
            "status": "approved",
            "message": f"Response approved and added to training data",
            "training_examples": len(TRAINING_DATA),
            "action_stored": request.action,
            "duplicate": False
        }
        
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e), "duplicate": False}
    finally:
        db.close()

@app.get("/stats")
async def get_stats():
    """Get training data statistics with updated info"""
    action_counts = {}
    
    for example in TRAINING_DATA:
        action = example.get('action', 'unknown')
        action_counts[action] = action_counts.get(action, 0) + 1
    
    return {
        "total_training_examples": len(TRAINING_DATA),
        "action_distribution": action_counts,
        "key_features": {
            "centralized_rules": "Single source of truth for all business logic",
            "phone_number": "(844) 679-1188 ONLY for hesitation/contact requests/confusion",
            "website_ctas": "High-intent prospects only",
            "positioning": "Completely online loan facilitators",
            "analysis": "Case-by-case lease analysis, not generic claims"
        },
        "supported_actions": {
            "respond": "Generate helpful response using centralized rules",
            "react": "Add thumbs up or heart reaction", 
            "delete": "Remove spam/accusations/hostility",
            "leave_alone": "Ignore harmless off-topic comments"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
