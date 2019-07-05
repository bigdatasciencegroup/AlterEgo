from twilio.rest import Client

account_sid = 'AC0e398957f7bb623ef1c7e7201c475dc8'
auth_token = '9772b33ec3152819b4f41ea9c75062e3'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="AlterEgo test message",
                     from_='+18573990716',
                     to='+16178395035'
                 )

print(message.sid)