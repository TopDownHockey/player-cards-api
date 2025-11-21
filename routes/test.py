from flask import jsonify

def test_route():
    return jsonify({"message": "Hello World"})

