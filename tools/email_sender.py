import yagmail
from tools.print_time_info import print_time_info


def send_email(mail,):
    mail_default = {
        'to': "achark@outlook.com",
        'subject': "Hello world!",
        'contents': 'This is a test message.',
        # 'attachments': 'document.pdf',
    }
    for key in mail_default:
        if key not in mail:
            raise TypeError('A mail should contain %s.' % (key,))

    yag = yagmail.SMTP(user="codemessager@gmail.com", password='Iamcodemessager.')
    yag.send(**mail)
    print_time_info('The message to %s has been sent successfully.' % mail['to'])