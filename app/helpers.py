import config
import os
import sendgrid
from sendgrid.helpers.mail import *



def emailNotification(emails, subject, message):
    sg = sendgrid.SendGridAPIClient(config.api['sendgrid']['key'])
    from_email = Email(config.api['sendgrid']['from_email'])
    to_email = [To(email) for email in emails]
    content = Content("text/plain", message)
    mail = Mail(from_email, to_email, subject, content)
    response = sg.client.mail.send.post(request_body=mail.get())
    return response.status_code
    #DEBUG
    # print(response.body)
    # print(response.headers)
    #END DEBUG