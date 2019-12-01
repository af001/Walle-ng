import requests
import json
import datetime
from ie_module import Module

class FaceNotification(Module):
    UNKNOWN_ID = -1

    def __init__(self, url, api_key):
        super(FaceNotification, self).__init__(url, api_key)
        self.url = url
        self.api_key = api_key
        self.faces_database = None
        self.notified = False
        self.now = datetime.datetime.now()
        self.known_identities = list()
        self.unknown_identities = list()

    def send_notification(self, msg):
        # Data to send to API Gateway
        data = {'message': msg.capitalize()} 
        headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}

        # sending post request and saving response as response object 
        r = requests.post(url=self.url, data=json.dumps(data), headers=headers) 
        
        if r.status_code == 200:
            print('[DEBUG] Recived: {}'.format(r.status_code))
        
        self.notified = True
        self.now = datetime.datetime.now()
        self.known_identities = list()
        self.unknown_identities = list()

    def build_message(self):
        if len(self.known_identities) > 0:
            message = 'Observed {}'.format(','.join(self.known_identities))
        else:
            message = 'Observed {} unidentified person(s)'.format(max(self.unknown_identities))
        print('[DEBUG] {}'.format(message))
        return message

    def preprocess(self, face_identities, unknowns):

        if len(face_identities) > 0:
            for face in face_identities:
                if not face.id == self.UNKNOWN_ID:
                    name = self.get_identity_label(face.id)
                    self.known_identities.append(name)
        if len(unknowns) > 0:
            self.unknown_identities.append(len(unknowns))

    def start_async(self, face_identities, unknowns):
        now = datetime.datetime.now()
        if self.now + datetime.timedelta(minutes = 1) < now:
            self.notified = False
        if not self.notified:
            self.preprocess(face_identities, unknowns)
            if len(self.known_identities) > 0 or len(self.unknown_identities) > 20:
                msg = self.build_message()
                self.send_notification(msg)
    
    def set_faces_database(self, database):
        self.faces_database = database

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label
