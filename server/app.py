
# This file is part of Imagine server.

# Imagine server is free software: you can redistribute it and / or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# Imagine server is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# A copy of the GNU General Public License is provided in the "COPYING" file. If not, see https:// www.gnu.org/licenses/

import os
import argparse
import random
import json
import base64
import datetime
from flask import Flask, request, jsonify, send_file, abort, make_response, render_template
from flask_cors import CORS, cross_origin

from celery import exceptions as celery_exceptions

from redis import Redis


from auth import get_tokens, auth_level, can_generate_tokens, generate_tokens, consume_token

parser = argparse.ArgumentParser()
parser.add_argument('--mock', action='store_true')
args = parser.parse_known_args()
mock = args[0].mock


if mock:
    from mock.tasks import txt2img, inpaint, cel_app
else:
    from tasks import txt2img, inpaint, cel_app

app  = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

imagine_db = Redis(host='localhost', db=3)

output_directory = "./out"
tmp_directory    = "./tmp"


import time

from datetime import date
global_prefix = int(time.time())
current_id = 1


guest_time_limit = 8
max_time_to_wait_for_answer = 14

os.makedirs(tmp_directory, exist_ok=True)





def queue_time_estimate():

    with cel_app.pool.acquire(block=True) as conn:
        tasks = conn.default_channel.client.lrange('celery', 0, -1)

    total_time = 0
    for task in tasks:
        j = json.loads(task)
        headers = j['headers']
        id = headers['id']
        total_time += float(imagine_db.get(id))
        #body = json.loads(base64.b64decode(j['body']))
        #decoded_tasks.append(body)

    return total_time



def time_estimate(action, params):
    if (action == "inpaint"):
        return guest_time_limit - 1  # FIXME
    elif (action == "txt2img"):
        area = params['W'] * params['H']
        steps_50_time = 3.2e-8 * area ** 1.5
        if area < 250000:
            steps_50_time = 4
        return steps_50_time * (params['steps'] / 50) * params['n_samples']
    raise Exception("Unknown action")
    




celery_checked_and_ready = False

@app.route("/")
def index():
    global celery_checked_and_ready
    if not celery_checked_and_ready:
        celery_db = Redis(host='localhost', db=1)
        if celery_db.exists("_kombu.binding.celery"):
            celery_checked_and_ready = True
    if celery_checked_and_ready:
        return render_template(
            'index.html',
            can_generate_tokens=can_generate_tokens(request),
            has_tokens=(get_tokens() is not None),
            queue_time=queue_time_estimate()
        )
    else:
        return "Server is starting... It should take less than 1 or 2 minutes. Refresh this page later."


@app.post("/access")
def show_tokens():
    if auth_level(request) != 'admin':
        return render_template('unauthorized.html'), 401

    return render_template('access.html', tokens=get_tokens(), usage=usage())


@app.post("/edit_tokens")
@cross_origin()
def set_tokens():
    if auth_level(request) != 'admin':
        return render_template('unauthorized.html'), 401

    tokens, admin_token = generate_tokens()

    return render_template('access.html', tokens=tokens, admin_token=admin_token, usage=usage())



@app.route("/benchmark", methods=['POST', 'GET'])
@cross_origin()
def benchmark():
    #return send_file('./mock/king.png', download_name='image.png')
    return jsonify({
        'ok': 200
    })



@app.get("/result")
@cross_origin()
def result():
    task_id = request.args.get('taskId')
    file_id = request.args.get('fileId')

    path = f"{output_directory}/{file_id}-0000.png"
    if mock:
        path = './mock/king.png'

    success = txt2img.AsyncResult(task_id).state == 'SUCCESS' or \
        inpaint.AsyncResult(task_id).state == 'SUCCESS'

    #FIXME
    if success and os.path.exists(path):
        reset_allowed_time(request)
        return send_file(path, download_name = 'image.png')
    else:
        abort(404)






locked_params = {
    "C" : 4,
    "f" : 8,
    "dyn" : None,
    "from_file": None,
    "n_rows" : 2,
    "plms" : False,
    "ddim_eta" : 0.0,
    "n_iter" : 1,
    "outdir" : output_directory,
    "skip_grid" : False,
    "skip_save" : True, #FIXME
    "fixed_code": False,
    "save_intermediate_every": 1000
}


@app.post("/run")
@cross_origin()
def run():

    level = auth_level(request)
    error = error_response_for_level(level)

    if (error):
        return error

    init_time = time.time()

    global current_id

    params = get_params(request)
    seed = params['seed']

    task_time = time_estimate('txt2img', params)

    
    if level == 'guest':
        if task_time > guest_time_limit:
            return job_too_long()
        if not allowed_to_use(request):
            return job_too_soon()


    params = {**params, **locked_params}
    params['file_prefix'] = f"{global_prefix}_{current_id}"

    handle_image_guide(request, params)

    current_id += 1

    time_in_queue = queue_time_estimate() + task_time

    record_usage(request, time_in_queue)

    task = txt2img.apply_async(priority=priority(level), kwargs={'params': params})
    consume_token(request)

    imagine_db.set(task.id, task_time)

    print(f'ETA : {time_in_queue}s')

    start_time = time.time()
    print(f'init time: {start_time - init_time}s')

    response_type = request.headers.get('Accept')

    if time_in_queue < max_time_to_wait_for_answer:
        try:
            task.get(timeout=max_time_to_wait_for_answer + 1)
            reset_allowed_time(request)
            path = f"{output_directory}/{params['file_prefix']}-0000.png"

            if mock:
                path = './mock/king.png'

            delta_time = time.time() - start_time

            
            
            delta_time = time.time() - start_time
            print(f'Generation time: {delta_time}s')

            return response_with_result(response_type, path, delta_time, seed)
        except celery_exceptions.TimeoutError:
            return timeout_error(response_type)
    else:
        return response_when_in_queue(
            response_type,
            task.id,
            params['file_prefix'],
            time_in_queue,
            seed
        )
        



def handle_image_guide(request, params):
    if 'imageGuide' in request.files and request.files['imageGuide'].filename != '':
        path = os.path.join(tmp_directory, f"{global_prefix}_{current_id}.png")
        request.files['imageGuide'].save(path)
        params['image_guide'] = path

        if 'maskForBlend' in request.files and \
                request.files['maskForBlend'].filename != '':
            mask_path = os.path.join(
                tmp_directory, f"{global_prefix}_{current_id}_mask.png")
            request.files['maskForBlend'].save(mask_path)
            params['blend_mask'] = mask_path
            params['mask_blur'] = max(request.form.get(
                'maskBlur',  default=10, type=int), 0)
        else:
            params['strength'] = max(min(request.form.get(
                'strength', default=0.5, type=float), 0.99), 0.01)

    else:
        params['image_guide'] = False


def get_params(request):
    form = request.form
    return {
        'prompt':     form.get('prompt',   default="countryside landscape, Trending on artstation."),
        'W':          min(form.get('width',    default=512, type=int), 2048),
        'H':          min(form.get('height',   default=512, type=int), 2048),
        'scale':      form.get('guidance',     default=7.0, type=float),
        'seed':       form.get('seed', default=random.randint(0, 99999999), type=int),
        'steps':      min(form.get('steps',    default=50, type=int), 150),
        'n_samples': min(form.get('samples',  default=1, type=int), 8),
        'blend_mask': None,
        'return_changes_only': form.get('returnChangesOnly', default=False, type=bool)
    }





@app.post("/inpaint")
@cross_origin()
def run_inpaint():

    level = auth_level(request)
    error = error_response_for_level(level)
    if (error is not None):
        return error

    global current_id

    if 'file' not in request.files:
        abort(422)

    file = request.files['file']
    if file.filename == '':
        abort(422)

    path = os.path.join(tmp_directory, f"{global_prefix}_{current_id}.png")

    params = {
        'image_path':  path,
        'outdir':      output_directory,
        'file_prefix': f"{global_prefix}_{current_id}"
    }

    task_time = time_estimate('inpaint', params)

    if level == 'guest':
        if task_time > guest_time_limit:
            return job_too_long()
        if not allowed_to_use(request):
            return job_too_soon()



    time_in_queue = queue_time_estimate() + task_time
    record_usage(request, time_in_queue)

    file.save(path)

    current_id += 1
    

    task = inpaint.apply_async(priority=priority(level), kwargs={'params': params})
    consume_token(request)

    imagine_db.set(task.id, task_time)
    
    response_type = request.headers.get('Accept')

    if time_in_queue < max_time_to_wait_for_answer:
        try:
            task.get(timeout=max_time_to_wait_for_answer + 1)
            reset_allowed_time(request)
            
            path = f"{output_directory}/{params['file_prefix']}-0000.png"
            if mock:
                path = './mock/king.png'

            response = make_response(send_file(path, download_name='image.png'))
            return response
        except celery_exceptions.TimeoutError:
            return timeout_error(response_type)
    else:
        return response_when_in_queue(
            response_type,
            task.id,
            params['file_prefix'],
            time_in_queue,
            None
        )
    




def priority(level):
    if level == 'user':
        return 5
    elif level == 'admin':
        return 0
    else:
        return 10


def ip_from_request(request):
    if 'X-Forwarded-For' in request.headers:
        ip_str = request.headers['X-Forwarded-For']
        return ip_str.split(',')[0]
    return request.remote_addr

def usage ():
    result = {}
    for key in imagine_db.scan_iter("usage_count_*"):
        ip = str(key, 'utf-8').split('_')[2]
        time = imagine_db.get(f'next_allowed_time_{ip}')
        next_time = datetime.datetime.fromtimestamp(
            int(float(time))) if time else None
        result[ip] = {
            'count': int(imagine_db.get(key)),
            'next_allowed_time': next_time
        }
    return result


def record_usage(request, seconds):
    ip = ip_from_request(request)
    time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    imagine_db.incr(f'usage_count_{ip}')
    imagine_db.set(f'next_allowed_time_{ip}', time.timestamp())



def reset_allowed_time(request):
    ip = ip_from_request(request)
    time = datetime.datetime.now()
    imagine_db.set(f'next_allowed_time_{ip}', time.timestamp())


def usage_count_for_ip(request):
    ip = ip_from_request(request)
    return imagine_db.get(f'usage_count_{ip}')


def next_allowed_time(request):
    ip = ip_from_request(request)
    time = imagine_db.get(f'next_allowed_time_{ip}')
    return time and datetime.datetime.fromtimestamp(float(time))


def allowed_to_use(request):
    time = next_allowed_time(request)
    if time is None:
        return True
    else:
        return time < datetime.datetime.now()



def response_with_result(response_type, path, delta_time, seed):
    if response_type == 'image/png':
        response = make_response(send_file(path, download_name='image.png'))
        response.headers['Access-Control-Expose-Headers'] = 'Imagine-Seed'
        response.headers['Imagine-Seed'] = seed
        return response
    else:
        return jsonify({
            'generationTime': delta_time,
            'seed': seed,
            'ok': 200,
            'image': base64.b64encode(open(path, 'rb').read()).decode('utf-8')
        })


def response_when_in_queue(response_type, task_id, file_id, time_in_queue, seed):
    if response_type == 'image/png':
        response = make_response(send_file('./1x1.png', mimetype='image/png'))
        response.headers['Imagine-Seed'] = seed
        response.headers['Imagine-Taskid'] = task_id
        response.headers['Imagine-Fileid'] = file_id
        response.headers['Access-Control-Expose-Headers'] = 'Imagine-Seed, Imagine-Taskid, Imagine-Fileid'
        return response
    else:
        params = {
            'status': 'queued',
            'estimatedTimeOfArrival': time_in_queue,
            'taskId': task_id,
            'fileId': file_id
        }

        if (seed is not None):
            params['seed'] = seed
        
        return jsonify(params)


def error_response_for_level(level):
    if level == 'quota_expired':
        return quota_expired(), 401
    elif level == 'unauthenticated':
        return unauthenticated(), 401
    elif level == 'wrong_auth':
        return wrong_auth(), 401
    else:
        return False



def unauthenticated(response_type):
    return error_response(response_type, 'unauthenticated')

def wrong_auth(response_type):
    return error_response(response_type, 'authentication failed')

def quota_expired(response_type):
    return error_response(response_type, 'quota expired')

def job_too_long(response_type):
    return error_response(response_type, 'job too long for a guest')

def job_too_soon(response_type):
    return error_response(response_type, 'job too soon for a guest')

def timeout_error(response_type):
    return error_response(response_type, 'timeout error')


def error_response(response_type, text):
    if response_type == 'image/png':
        response = make_response(send_file('./1x1.png', mimetype='image/png'))
        response.headers['Imagine-Error'] = text
        response.headers['Access-Control-Expose-Headers'] = 'Imagine-Error'
    else:
        response = jsonify({'error': text})
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3754, debug=True)






