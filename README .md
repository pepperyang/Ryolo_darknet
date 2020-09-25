# darknet-Ryolo-segmentation for Windows and Linux
This project is based on darknet to get image segmentation and rotational object detection

1、"make -j8"  or  build with vs2019   

2、	------to train on your own data

	"./darknet segmenter train the/path/to/data the/paht/to/cfg/file the/path/to/weights/files"

	"./darknet detector train_Ryolo the/path/to/data the/paht/to/cfg/file the/path/to/weights/files"

3、------test your trained network

	"./darknet segmenter test the/path/to/cfg/data the/path/to/cfg/file the/path/to/weights/files the/path/to/image " 
	
	./darknet detector test_Ryolo the/path/to/data the/path/to/cfg/file the/path/to/weights/files the/path/to/weights/files -ext_output 




