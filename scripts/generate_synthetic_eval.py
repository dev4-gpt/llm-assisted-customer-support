import json
import random
import uuid
from pathlib import Path

# Taxonomies
CATEGORIES = [
    "billing", "technical_bug", "authentication", 
    "feature_request", "general_inquiry", "cancellation"
]
PRIORITIES = ["low", "medium", "high", "critical"]

# Templates for Triage
TRIAGE_TEMPLATES = [
    ("I cannot access my {service}. It says {error}.", "authentication", "high"),
    ("My latest invoice for {amount} is incorrect. I was overcharged.", "billing", "critical"),
    ("Please add a {feature} to the {platform}.", "feature_request", "low"),
    ("How do I change my {setting}?", "general_inquiry", "low"),
    ("The {component} is completely broken and my team is blocked.", "technical_bug", "critical"),
    ("I want to cancel my subscription immediately.", "cancellation", "medium"),
    ("When I click on {component}, I get a 500 error.", "technical_bug", "high"),
    ("I need a refund for the {amount} charge on my card.", "billing", "high")
]

# Random fill-ins
SERVICES = ["account", "dashboard", "portal", "mobile app"]
ERRORS = ["invalid password", "account locked", "session expired", "access denied"]
AMOUNTS = ["$9.99", "$49.99", "$199.00", "$500"]
FEATURES = ["dark mode", "export to PDF", "CSV import", "2FA"]
PLATFORMS = ["iOS app", "web app", "desktop client", "API"]
SETTINGS = ["profile picture", "email address", "notification preferences", "timezone"]
COMPONENTS = ["checkout page", "login screen", "reports tab", "settings page"]

# Templates for Quality
QUALITY_TEMPLATES = [
    (
        "I was charged twice!", 
        "I'm sorry about the double charge. I have refunded {amount} to your card."
    ),
    (
        "The app keeps crashing.", 
        "Please reinstall the app and clear your cache."
    ),
    (
        "I want to cancel my account.", 
        "I'm sorry to see you go. Your account has been canceled."
    ),
    (
        "How do I reset my password?", 
        "You can click the 'Forgot Password' link on the login page."
    )
]

# Templates for Summarize
SUMMARIZE_TEMPLATES = [
    (
        "I need a refund.", 
        "I have processed your refund.", 
        "Customer requested a refund. Agent processed the refund successfully."
    ),
    (
        "My app is freezing on the dashboard.", 
        "Try updating to the latest version.", 
        "Customer reported app freezing. Agent advised updating the app."
    ),
    (
        "Can I add more users to my plan?", 
        "Yes, you can upgrade your plan in settings.", 
        "Customer asked about adding users. Agent explained how to upgrade the plan."
    )
]

def generate_triage():
    template, cat, prio = random.choice(TRIAGE_TEMPLATES)
    text = template.format(
        service=random.choice(SERVICES),
        error=random.choice(ERRORS),
        amount=random.choice(AMOUNTS),
        feature=random.choice(FEATURES),
        platform=random.choice(PLATFORMS),
        setting=random.choice(SETTINGS),
        component=random.choice(COMPONENTS)
    )
    # Add some noise
    if random.random() > 0.5:
        text = text + " " + random.choice(["Please help!", "Thanks.", "Fix this asap.", "Very urgent."])
    
    return {
        "task": "triage",
        "id": f"t_{uuid.uuid4().hex[:8]}",
        "ticket_text": text,
        "gold_priority": prio,
        "gold_category": cat
    }

def generate_quality():
    cust, agnt = random.choice(QUALITY_TEMPLATES)
    agnt = agnt.format(amount=random.choice(AMOUNTS))
    return {
        "task": "quality",
        "id": f"q_{uuid.uuid4().hex[:8]}",
        "ticket_text": cust,
        "agent_response": agnt
    }

def generate_summarize():
    cust, agnt, summ = random.choice(SUMMARIZE_TEMPLATES)
    return {
        "task": "summarize",
        "id": f"s_{uuid.uuid4().hex[:8]}",
        "turns": [
            {"role": "customer", "content": cust},
            {"role": "agent", "content": agnt}
        ],
        "gold_summary": summ
    }

def main():
    output_path = Path("data/golden/eval_set.jsonl")
    
    # Read existing
    existing_lines = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing_lines = f.readlines()
    
    # We want to add exactly 1000 lines
    new_records = []
    for _ in range(334):
        new_records.append(generate_triage())
    for _ in range(333):
        new_records.append(generate_quality())
    for _ in range(333):
        new_records.append(generate_summarize())
        
    # Shuffle the new records
    random.shuffle(new_records)
    
    with open(output_path, "a") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully appended {len(new_records)} synthetic records to {output_path}.")
    print(f"Total records is now {len(existing_lines) + len(new_records)}.")

if __name__ == "__main__":
    main()
