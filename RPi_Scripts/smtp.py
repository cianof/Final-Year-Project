import smtplib

def sendemail():
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login("fyp.rpi.email@gmail.com", "CiansRPi21")
    msg = "Help, I have had an accident and require assistance, Your Name Here"
    server.sendmail("fyp.rpi.email@gmail.com", "target@email.com", msg)
    server.quit()
    
sendemail()