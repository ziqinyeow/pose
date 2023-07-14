clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

demo:
	python demo.py

test:
	python test.py

onnx:
	python onnx.py test/mmdeploy/rtmpose-ort/rtmpose-m/end2end.onnx data/human-pose.jpg
	