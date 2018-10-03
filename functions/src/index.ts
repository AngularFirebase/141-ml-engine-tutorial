import * as functions from 'firebase-functions';

import { google } from 'googleapis';
const ml = google.ml('v1')

export const predictHappiness = functions.https.onRequest(async (request, response) => {
    
    const instances = request.body.instances; 
    const model = request.body.model; 

    const { credential } = await google.auth.getApplicationDefault();
    const modelName = `projects/angularfirebase-267db/models/${model}`;

    const preds = await ml.projects.predict({
        auth: credential,
        name: modelName,
        requestBody: { 
            instances
        }
    } as any);

    response.send(JSON.stringify(preds.data))

});
