import os
import json
from dotenv import load_dotenv
from arcadepy import Arcade
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API keys and user ID from environment variables
ARCADE_API_KEY = os.getenv("ARCADE_API_KEY")
ARCADE_USER_ID = os.getenv("ARCADE_USER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
arcade_client = Arcade(api_key=ARCADE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Function to analyze email and extract meeting details in a single API call
def analyze_email(email_body, email_subject, sender):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You analyze emails to determine if they contain meeting or appointment requests, or any indication that the sender wants to connect, talk, or have a conversation. Be sensitive to implicit requests for time or discussions, not just explicit meeting requests."},
                {"role": "user", "content": f"Sender: {sender}\nSubject: {email_subject}\n\nBody: {email_body}"}
            ],
            functions=[
                {
                    "name": "analyze_meeting_request",
                    "description": "Analyze if this is a meeting request and extract relevant details",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_meeting_request": {
                                "type": "boolean",
                                "description": "Whether this email contains a meeting or appointment request or any indication of wanting to connect"
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Confidence level in the meeting request determination"
                            },
                            "first_name": {
                                "type": "string",
                                "description": "The first name of the sender"
                            },
                            "purpose": {
                                "type": "string",
                                "description": "The purpose or topic of the meeting"
                            }
                        },
                        "required": ["is_meeting_request", "confidence", "first_name"]
                    }
                }
            ],
            function_call={"name": "analyze_meeting_request"}
        )
        
        function_call = response.choices[0].message.function_call
        if function_call:
            result = json.loads(function_call.arguments)
            is_meeting_request = result.get('is_meeting_request', False)
            confidence = result.get('confidence', 'low')
            
            # Print the analysis result for debugging
            print(f"Meeting request analysis: {is_meeting_request} (confidence: {confidence})")
            
            return result
        return {"is_meeting_request": False, "first_name": "there", "confidence": "low"}  # Fallback
    except Exception as e:
        print(f"Error analyzing email: {e}")
        return {"is_meeting_request": False, "first_name": "there", "confidence": "low"}

# Function to generate a response to a meeting request
def generate_meeting_response(meeting_details):
    try:
        first_name = meeting_details.get("first_name", "there")
        purpose = meeting_details.get("purpose", "meeting")
        
        response = f"Hi {first_name},\n\nThank you for your email about the {purpose}. I'd be happy to meet with you. You can view my availability and schedule a time in one of two ways:\n\n1. Book directly through my Calendly: https://calendly.com/sattensil/new-meeting\n2. Check my calendar availability and send me a calendar invitation for a time that works for both of us\n\nLooking forward to our conversation!\n\nBest regards,\nScarlett"
        
        return response
    except Exception as e:
        print(f"Error generating meeting response: {e}")
        return "Hi there,\n\nThank you for your email. I'd be happy to meet with you. You can view my availability and schedule a time in one of two ways:\n\n1. Book directly through my Calendly: https://calendly.com/sattensil/new-meeting\n2. Check my calendar availability and send me a calendar invitation for a time that works for both of us\n\nLooking forward to our conversation!\n\nBest regards,\nScarlett"

# Authorize the tool
auth_response = arcade_client.tools.authorize(
    tool_name="Google.ListEmails@1.2.1",
    user_id=ARCADE_USER_ID,
)

# Check if authorization is completed
if auth_response.status != "completed":
    print(f"Click this link to authorize: {auth_response.url}")

# Wait for the authorization to complete
auth_response = arcade_client.auth.wait_for_completion(auth_response)

if auth_response.status != "completed":
    raise Exception("Authorization failed")

print("🚀 Authorization successful!")

# Get the last 30 emails
result = arcade_client.tools.execute(
    tool_name="Google.ListEmails@1.2.1",
    input={
        "owner": "ArcadeAI",
        "name": "arcade-ai",
        "n_emails": "30"
    },
    user_id=ARCADE_USER_ID,
)

# Print all emails and their labels for debugging
print("\n===== ALL EMAILS =====")
if result.output and result.output.value and 'emails' in result.output.value:
    print(f"Total emails retrieved: {len(result.output.value['emails'])}")
    for i, email in enumerate(result.output.value['emails']):
        print(f"\nEmail #{i+1}:")
        print(f"  ID: {email.get('id')}")
        print(f"  Subject: {email.get('subject', 'No subject')}")
        print(f"  From: {email.get('from', 'Unknown sender')}")
        print(f"  Labels: {email.get('label_ids', [])}")

# Filter to unread emails in INBOX
unread_inbox_emails = []
if result.output and result.output.value and 'emails' in result.output.value:
    for email in result.output.value['emails']:
        label_ids = email.get('label_ids', [])
        if 'UNREAD' in label_ids and 'INBOX' in label_ids:
            unread_inbox_emails.append(email)
    
    print(f"\nFound {len(unread_inbox_emails)} unread emails in INBOX")
    
    # Print details of unread inbox emails
    for i, email in enumerate(unread_inbox_emails):
        print(f"\nUnread INBOX Email #{i+1}:")
        print(f"  ID: {email.get('id')}")
        print(f"  Subject: {email.get('subject', 'No subject')}")
        print(f"  From: {email.get('from', 'Unknown sender')}")

# Process each unread inbox email
meeting_request_emails = []
for email in unread_inbox_emails:
    email_id = email.get('id')
    subject = email.get('subject', '')
    body = email.get('body', '')
    sender = email.get('from', '')
    
    # Analyze email for meeting request and extract details
    analysis_result = analyze_email(body, subject, sender)
    is_meeting_request = analysis_result.get('is_meeting_request', False)
    confidence = analysis_result.get('confidence', 'low')
    
    # Only consider high or medium confidence meeting requests
    if is_meeting_request and confidence in ['high', 'medium']:
        meeting_request_emails.append({
            'email_id': email_id,
            'sender': sender,
            'meeting_details': analysis_result
        })

print(f"Found {len(meeting_request_emails)} emails with meeting requests")

# Respond to each meeting request
for meeting_email in meeting_request_emails:
    email_id = meeting_email['email_id']
    meeting_details = meeting_email['meeting_details']
    
    # Generate response
    response_body = generate_meeting_response(meeting_details)
    
    # Send the response
    response = arcade_client.tools.execute(
        tool_name="Google.ReplyToEmail@1.2.1",
        input={
            "owner": "ArcadeAI",
            "name": "arcade-ai",
            "body": response_body,
            "reply_to_message_id": email_id,
            "reply_to_whom": "only_the_sender"
        },
        user_id=ARCADE_USER_ID,
    )
    
    # Mark the email as read by removing the UNREAD label
    mark_read_result = arcade_client.tools.execute(
        tool_name="Google.ChangeEmailLabels@1.2.1",
        input={
            "owner": "ArcadeAI",
            "name": "arcade-ai",
            "email_id": email_id,
            "labels_to_remove": "UNREAD"
        },
        user_id=ARCADE_USER_ID,
    )
    
    if mark_read_result.status == "success":
        print(f"Marked email as read: {email_id}")
    else:
        print(f"Failed to mark email as read: {email_id}")

    print(f"Responded to meeting request from {meeting_email['sender']}")
# Print summary of actions
print("\n===== SUMMARY =====")
print(f"Total emails checked: {len(result.output.value.get('emails', []))}")
print(f"Unread emails in INBOX: {len(unread_inbox_emails)}")
print(f"Meeting requests identified: {len(meeting_request_emails)}")
print(f"Responses sent: {len(meeting_request_emails)}")
print(f"Emails marked as read: {len(meeting_request_emails)}")