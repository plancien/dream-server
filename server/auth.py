# This file is part of Imagine server.

# Imagine server is free software: you can redistribute it and / or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# Imagine server is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# A copy of the GNU General Public License is provided in the "COPYING" file. If not, see https:// www.gnu.org/licenses/

import json
import string
import random

from redis import Redis

imagine_db = Redis(host='localhost', db=3)

def get_tokens():
    tokens = imagine_db.get('tokens')
    return tokens and json.loads(tokens)


def auth_level_for_token(token):
    tokens = get_tokens()

    if not tokens:
        return 'admin'

    if token in tokens:
        return tokens[token]
    else:
        return 'guest'

def can_generate_tokens(request):
    return auth_level(request) == 'admin'


def random_token(length=10):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join((random.choice(letters_and_digits)
                    for i in range(length)))


def generate_tokens():
    tokens = {}
    tokens[random_token()] = 'user'

    admin_token = random_token()
    tokens[admin_token] = 'admin'

    imagine_db.set('tokens', json.dumps(tokens))
    return tokens, admin_token


def auth_level(request):
    token = request.form.get('token')
    return auth_level_for_token(token)


def consume_token(request):
    return True