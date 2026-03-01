
import smtplib
from email.mime.text import MIMEText
import os
from config import settings

def send_email(content):

    sender = settings.EMAIL_USER
    password = settings.EMAIL_PASS
    receiver = settings.EMAIL_TO


    msg = MIMEText(content)
    msg["Subject"] = "AI News Intelligence Report"
    msg["From"] = sender
    msg["To"] = receiver

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

    print("Email sent successfully.")