import boto3
import urllib.parse
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

TOPIC_ARN_SMS = "<sns_topic_arn>"
SMS_SUBJ = "Alert"
SMS_MESG = "Observed unknown person(s)"

'''Setup SNS resource'''
def publish_sms_msg(topic_arn=TOPIC_ARN_SMS, sms_mesg=SMS_MESG, sms_subj=SMS_SUBJ):
    sns = boto3.resource('sns')
    publish_sms(sns, topic_arn, sms_mesg, sms_subj)

'''Send the SMS'''
def publish_sms(sns, topic_arn, sms_mesg, sms_subj):
    topic = sns.Topic(topic_arn)
    topic.publish(Message=sms_mesg, Subject=sms_subj)

'''Event handler'''
def lambda_handler(event, context):
    if event['httpMethod'] == 'POST':
        msg = event['body']
        msg = urllib.parse.parse_qs(msg)
        if msg['message']:
            alert = msg['message'][0]
            logger.info(alert)
            publish_sms_msg(sms_mesg=alert)
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": "OK"
            }
        else:
            return {
                "statusCode": 499,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": "Go Away!"
            }
    else:
        return {
            "statusCode": 499,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": "Go Away!"
        }
