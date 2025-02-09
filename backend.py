from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
from datetime import datetime, timedelta
from agents.orchestrator import orchestrator_agent  
from flask_jwt_extended import decode_token
import json
import random
import traceback  # For debugging
app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///xfinitychatbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "supersecretkey"  
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
db = SQLAlchemy(app)
jwt = JWTManager(app)
class User(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   name = db.Column(db.String(100), nullable=False)
   email = db.Column(db.String(100), unique=True, nullable=False)
   password_hash = db.Column(db.String(200), nullable=False)
   created_at = db.Column(db.DateTime, default=datetime.utcnow)
class QueryLog(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   user_name = db.Column(db.String(100), nullable=False)
   email = db.Column(db.String(100), nullable=False)
   timestamp = db.Column(db.DateTime, default=datetime.utcnow)
   message = db.Column(db.Text, nullable=False)
   query_category = db.Column(db.String(50), nullable=False)
   sentiment = db.Column(db.String(50), nullable=False)
   response = db.Column(db.Text, nullable=False)
class ComplaintLog(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   user_name = db.Column(db.String(100), nullable=False)
   email = db.Column(db.String(100), nullable=False)
   timestamp = db.Column(db.DateTime, default=datetime.utcnow)
   complaint = db.Column(db.Text, nullable=False)
   sentiment = db.Column(db.String(50), nullable=False)
class FeedbackLog(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   user_name = db.Column(db.String(100), nullable=False)
   email = db.Column(db.String(100), nullable=False)
   timestamp = db.Column(db.DateTime, default=datetime.utcnow)
   feedback = db.Column(db.Text, nullable=False)
   sentiment = db.Column(db.String(50), nullable=False)
@app.route("/")
def home():
   return "<h2> Backend is up and running :) </h2>"
#session_data = {"needs_confirmation": False}

@app.route("/signup", methods=["POST"])
def signup():
  data = request.json
  name = data.get("name")
  email = data.get("email")
  password = data.get("password")
  if User.query.filter_by(email=email).first():
      return jsonify({"error": "Email already exists"}), 400
  hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
  new_user = User(name=name, email=email, password_hash=hashed_password)
  db.session.add(new_user)
  db.session.commit()
  return jsonify({"message": "User registered successfully!"}), 201

@app.route("/login", methods=["POST"])
def login():
  data = request.json
  email = data.get("email")
  password = data.get("password")
  user = User.query.filter_by(email=email).first()
  if not user or not bcrypt.check_password_hash(user.password_hash, password):
      return jsonify({"error": "Invalid credentials"}), 401
  user_identity = json.dumps({"name": user.name, "email": user.email})
  access_token = create_access_token(identity=user_identity, expires_delta=timedelta(days=1))
  return jsonify({"access_token": access_token, "name": user.name, "email": user.email})

@app.route("/check_auth", methods=["GET"])
@jwt_required()
def check_auth():
  user = get_jwt_identity()
  return jsonify(user)


session_data = {
 "needs_confirmation": False
}


@app.route('/process_message', methods=['POST'])
def process_message():
   data = request.json
   user_message = data.get("message", "").strip().lower()
   token = request.headers.get("Authorization")
   if token and token.startswith("Bearer "):
      token = token.split(" ")[1]
   user_name, email = None, None
   try:
      if token:
          decoded_token = decode_token(token)
          user_identity = json.loads(decoded_token["sub"])
          user_name = user_identity["name"]
          email = user_identity["email"]
      else:
          raise Exception("No token provided")
   except Exception:
      user_name = f"GuestUser{random.randint(100, 999)}"
      email = "guest@example.com"
   try:
       # **Handle confirmation first, if required**
       if session_data["needs_confirmation"]:
           result = orchestrator_agent.invoke({
               "user_message": user_message,
               "sentiment": "",
               "category": "",
               "response": "",
               "xfinity_related": "",
               "harmful_content": "",
               "contains_sensitive_info": "",
               "buttons": [],
               "needs_confirmation": True
           })
           session_data["needs_confirmation"] = result.get("needs_confirmation", False)
           return jsonify({
               "response": result["response"],
               "buttons": result.get("buttons", []),
               "needs_confirmation": session_data["needs_confirmation"]
           })
       # **Otherwise, process a new message normally**
       result = orchestrator_agent.invoke({
           "user_message": user_message,
           "sentiment": "",
           "category": "",
           "xfinity_related": "",
           "harmful_content": "",
           "contains_sensitive_info": "",
           "response": "",
           "buttons": [],
           "needs_confirmation": False
       })
       session_data["needs_confirmation"] = result.get("needs_confirmation", False)
       category = result.get("category")
       sentiment = result.get("sentiment")
       response = result["response"]
       harmful_content = result.get("harmful_content", "No")
       contains_sensitive_info = result.get("contains_sensitive_info", "No")
       xfinity_related = result.get("xfinity_related", "Yes")
       
       if harmful_content == "No" and contains_sensitive_info == "No" and xfinity_related == "Yes":
           if category == "General Query":
               if not session_data["needs_confirmation"]:  # ✅ Log only after confirmation is completed
                   query_log = QueryLog(
                       user_name=user_name,
                       email=email,
                       message=user_message,
                       query_category=category,
                       sentiment=sentiment,
                       response=response
                   )
                   db.session.add(query_log)
           elif category == "Complaint":
               if email == "guest@example.com":
                   return jsonify({
                       "response": "Please log in to submit complaints.",
                       "buttons": [{"label": "Sign In", "action": "signin"}],
                       "needs_confirmation": False
                   })
               complaint_log = ComplaintLog(
                   user_name=user_name,
                   email=email,
                   complaint=user_message,
                   sentiment=sentiment
               )
               db.session.add(complaint_log)
           elif category == "Feedback":
               feedback_log = FeedbackLog(
                   user_name=user_name,
                   email=email,
                   feedback=user_message,
                   sentiment=sentiment
               )
               db.session.add(feedback_log)
           db.session.commit()
       return jsonify({
           "response": response,
           "buttons": result.get("buttons", []),
           "needs_confirmation": session_data["needs_confirmation"]
       })
   except Exception as e:
       print("[ERROR] Failed to process message:", str(e))
       traceback.print_exc()
       return jsonify({"error": str(e)})
   
@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
   """Fetch last 10 chat queries from QueryLog"""
   chat_logs = QueryLog.query.order_by(QueryLog.timestamp.desc()).limit(10).all()
   chat_data = [
       {
           "user_name": log.user_name,
           "email": log.email,
           "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
           "message": log.message,
           "query_category": log.query_category,
           "sentiment": log.sentiment,
           "response": log.response
       }
       for log in chat_logs
   ]
   return jsonify({"chat_history": chat_data})

@app.route('/get_users', methods=['GET'])
def get_users():
   """Fetch all registered users"""
   users = User.query.all()
   user_data = [
       {
           "id": user.id,
           "name": user.name,
           "email": user.email,
           "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S")
       }
       for user in users
   ]
   return jsonify({"users": user_data})

@app.route('/get_feedbacks', methods=['GET'])
def get_feedbacks():
   """Fetch all feedback entries"""
   feedbacks = FeedbackLog.query.order_by(FeedbackLog.timestamp.desc()).all()
   feedback_data = [
       {
           "user_name": feedback.user_name,
           "email": feedback.email,
           "timestamp": feedback.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
           "feedback": feedback.feedback,
           "sentiment": feedback.sentiment
       }
       for feedback in feedbacks
   ]
   return jsonify({"feedbacks": feedback_data})

@app.route('/get_complaints', methods=['GET'])
def get_complaints():
   """Fetch all complaints"""
   complaints = ComplaintLog.query.order_by(ComplaintLog.timestamp.desc()).all()
   complaint_data = [
       {
           "user_name": complaint.user_name,
           "email": complaint.email,
           "timestamp": complaint.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
           "complaint": complaint.complaint,
           "sentiment": complaint.sentiment
       }
       for complaint in complaints
   ]
   return jsonify({"complaints": complaint_data})

@app.route('/get_queries', methods=['GET'])
def get_queries():
   """Fetch all queries from QueryLog"""
   query_logs = QueryLog.query.order_by(QueryLog.timestamp.desc()).all()
   query_data = [
       {
           "user_name": log.user_name,
           "email": log.email,
           "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
           "message": log.message,
           "query_category": log.query_category,
           "sentiment": log.sentiment,
           "response": log.response
       }
       for log in query_logs
   ]
   return jsonify({"queries": query_data})

if __name__ == "__main__":
   with app.app_context():
       db.create_all()
   app.run(debug=True)