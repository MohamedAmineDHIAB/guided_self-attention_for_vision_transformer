'''
Loader class which uses the loader widget in juptyter
notebooks to import images easily

'''

import ipywidgets as widgets
from PIL import Image
import io

class Loader(object):
	def __init__(self):
		self.uploader = widgets.FileUpload(accept='image/*', multiple=False)
		self._start()

	def _start(self):
		display(self.uploader)

	def getLastImage(self):
		try:
			for uploaded_filename in self.uploader.value:
				uploaded_filename = uploaded_filename
			img = Image.open(io.BytesIO(bytes(self.uploader.value[uploaded_filename]['content'])))

			return img
		except:
			return None


	def saveImage(self, path):
		with open(path, 'wb') as output_file:
			for uploaded_filename in self.uploader.value:
				content = self.uploader.value[uploaded_filename]['content']
				output_file.write(content)