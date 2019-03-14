package main;

//import java.awt.Frame;
//import java.awt.event.WindowAdapter;
//import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.imageio.ImageIO;

import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.vector.Vector3f;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

//import com.jogamp.opengl.GLCapabilities;
//import com.jogamp.opengl.GLProfile;
//import com.jogamp.opengl.awt.GLCanvas;

import engineTester.PoseEstimationValues;
import entities.Camera;
import entities.Entity;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import main.utils.Utils;
import models.RawModel;
import models.TexturedModel;
import renderEngine.DisplayManager;
import renderEngine.Loader;
//import renderEngine.OBJLoader;
import renderEngine.Renderer;
import shaders.StaticShader;
import textures.ModelTexture;

public class ImageRecognition
{	
	private static final String PATH_IMAGES = "images/";
	
	private static final Scalar COLOR_BLUE = new Scalar(255, 200, 50);
	private static final Scalar COLOR_GREEN = new Scalar(0, 255, 0);
	private static final Scalar COLOR_RED = new Scalar(0, 0, 255);
	
	private static int cameraId = 0;
	
	// the FXML button
	@FXML
	private Button button;
	@SuppressWarnings("unused")
	private Button button2;
	// the FXML image view
	@FXML
	private ImageView currentFrame;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that realizes the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive = false;
	// the id of the camera to be used
	
	private static Mat img1 = Highgui.imread(PATH_IMAGES + "5house.png");//aruco
	private static Mat img2 = Highgui.imread(PATH_IMAGES + "6lena.png");//5house
	private static Mat img3 = Highgui.imread(PATH_IMAGES + "black.png");
	
	//private int countFrames = 0;

	private static Mat imgCurrent;	
	private String textImage;	
	private String textImageName;	
	private String textTranslation;
	private String textRotation;
	private int rotX, rotY, rotZ;
	private int transX, transY, transZ;
	@SuppressWarnings("unused")
	private Point matchLoc;
	
	private float nndrRatio = 1;
	private boolean drawMatches = false;
	private boolean drawProjectedCube = false;
	@FXML
    private Slider mySlider;
    @FXML
    private TextField textField;
	
	/**
	 * Get a frame from the opened video stream (if any)
	 *
	 * @return the {@link Mat} to show
	 */
	private Mat grabFrame()
	{
		// init everything
		Mat img = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(img);
				
				// if the frame is not empty, process it
				if (!img.empty())
				{		
					//imgCurrent = img1;
			    	textImage = "Objekt erkannt!";
			    	textImageName = "";
					//chessBoardAR(img);
			    	
					matchTemplate(img);
			    	//imgCurrent = img1;
					
					matchFeaturePoints(img, objectPoints3d, scenePoints2d);
					//testPoseEstimation(objectPoints3d, scenePoints2d);
					drawInfosOnFrame(img);
				}				
			}
			catch (Exception e)
			{
				// log the error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return img;
	}

    public void initialize() {
    	nndrRatio = 0.75f;
    	mySlider.setValue(nndrRatio);

        mySlider.valueProperty().addListener((observable, oldValue, newValue) -> {
            nndrRatio = ((Double) newValue).floatValue();            
        });
    }
    
	public static void testOpenGL() {

		DisplayManager.createDisplay();

		Loader loader = new Loader();
		StaticShader shader = new StaticShader();
		Renderer renderer = new Renderer(shader);
/*
		float[] vertices = { 
				// left bottom triangle
				-0.6f, 0.5f, 0f,
				-0.5f, -0.5f, 0f, 
				0.5f, -0.5f, 0f, 
				// right top triangle
				0.5f, -0.5f, 0f, 
				0.5f, 0.5f, 0f, 
				-0.5f, 0.5f, 0f 
		};
*/
/*		
		float[] vertices = { 
				-0.5f, 0.5f, 0f,
				-0.5f, -0.5f, 0f, 
				0.5f, -0.5f, 0f, 
				0.5f, 0.5f, 0f
		};
		
		int[] indices = {
				0,1,3,
				3,1,2
		};
		
		float[] textureCoord = {
				0,0,
				0,1,
				1,1,
				1,0
		};
*/		
//*		
		float[] vertices = {			
				-0.5f,0.5f,-0.5f,	
				-0.5f,-0.5f,-0.5f,	
				0.5f,-0.5f,-0.5f,	
				0.5f,0.5f,-0.5f,		
				
				-0.5f,0.5f,0.5f,	
				-0.5f,-0.5f,0.5f,	
				0.5f,-0.5f,0.5f,	
				0.5f,0.5f,0.5f,
				
				0.5f,0.5f,-0.5f,	
				0.5f,-0.5f,-0.5f,	
				0.5f,-0.5f,0.5f,	
				0.5f,0.5f,0.5f,
				
				-0.5f,0.5f,-0.5f,	
				-0.5f,-0.5f,-0.5f,	
				-0.5f,-0.5f,0.5f,	
				-0.5f,0.5f,0.5f,
				
				-0.5f,0.5f,0.5f,
				-0.5f,0.5f,-0.5f,
				0.5f,0.5f,-0.5f,
				0.5f,0.5f,0.5f,
				
				-0.5f,-0.5f,0.5f,
				-0.5f,-0.5f,-0.5f,
				0.5f,-0.5f,-0.5f,
				0.5f,-0.5f,0.5f				
		};
		
		int[] indices = {
				0,1,3,	
				3,1,2,	
				
				4,5,7,
				7,5,6,
				
				8,9,11,
				11,9,10,
				
				12,13,15,
				15,13,14,
				
				16,17,19,
				19,17,18,
				
				20,21,23,
				23,21,22
		};
		
		float[] textureCoords = {
				//back
				1,0.5f,
				1,1,
				0.5f,1,
				0.5f,0.5f,
				//front
				0.5f,0,
				0.5f,0.5f,
				1,0.5f,
				1,0,	
				//right
				0.5f,0.5f,
				0.5f,1,	
				0,1,
				0,0.5f,			
				//left
				0,0,
				0,0.5f,
				0.5f,0.5f,
				0.5f,0,
				//above
				0,0.5f,
				0,0,
				0.5f,0,
				0.5f,0.5f,
				//below
				0,0,
				0,1,
				1,1,
				1,0			
		};
//*/		
		RawModel model = loader.loadToVAO(vertices, textureCoords, indices);
		//RawModel model = OBJLoader.loadObjModel("cube3D", loader);
		TexturedModel staticModel = new TexturedModel(model,new ModelTexture(loader.loadTexture("cube3D")));
		Entity entity = new Entity(staticModel, new Vector3f(0,0,-2.5f),0,0,0,1);
		Camera camera = new Camera();
		
		while (!Display.isCloseRequested()) {
			//entity.increaseRotation(0.2f, 0.0f, 0.0f);
			//entity.setRotX(-45);

			if(imgCurrent == img1) {
				//entity.setPosition(new Vector3f(PoseEstimationValues.posX, -PoseEstimationValues.posY, -2.5f-PoseEstimationValues.posZ));
				
				entity.setRotX(PoseEstimationValues.rotX);
				entity.setRotY(PoseEstimationValues.rotY);
				
				camera.roll(PoseEstimationValues.rotZ);
			} else {
				entity.setPosition(new Vector3f(0, 0, -2.5f));
				
				entity.setRotX(PoseEstimationValues.rotX);
				entity.setRotY(90+PoseEstimationValues.rotY);
				
				camera.roll(PoseEstimationValues.rotZ);
			}
			//entity.setRotX(PoseEstimationValues.rotX);
			//entity.setRotY(PoseEstimationValues.rotY);
			//entity.setRotZ(PoseEstimationValues.rotZ);
			
			camera.move();
			renderer.prepare();
			shader.start();
			shader.loadViewMatrix(camera);
			renderer.render(entity,shader);
			shader.stop();
			DisplayManager.updateDisplay();
		}

		shader.cleanUp();
		loader.cleanUp();
		DisplayManager.closeDisplay();
		
		startPreview = true;
		/*
		GLProfile glp = GLProfile.getDefault();
		GLCapabilities caps = new GLCapabilities(glp);
		GLCanvas canvas = new GLCanvas(caps);
		
		Frame frame = new Frame();
		frame.setSize(500, 500);
		frame.add(canvas);
		frame.setVisible(true);
		
		frame.addWindowListener(new WindowAdapter() {
			public void WindowClosing(WindowEvent e) {
				System.out.println("test");
				System.exit(0);
			}
		});
		
		*/
	}
	
	private void drawInfosOnFrame(Mat img) {	
		Core.putText(img, textImage, new Point(50,50), 3, 1, COLOR_BLUE, 2);
		
		textRotation = "Rotation:   ";
		textRotation += " x=";
		textRotation += rotX;
		textRotation += " y=";
		textRotation += rotY;
		textRotation += " z=";
		textRotation += rotZ;
		
		Core.putText(img, textRotation, new Point(50,450), 3, 1, COLOR_RED, 2);
		
		textTranslation = "Translation:";
		textTranslation += " x=";
		textTranslation += transX;
		textTranslation += " y=";
		textTranslation += transY;
		textTranslation += " z=";
		textTranslation += transZ;
		Core.putText(img, textTranslation, new Point(50,400), 3, 1, COLOR_RED, 2);
		
		//double offsetX = drawMatches ? imgCurrent.cols() : 0;

		//if (matchLoc != null) 
			//Core.rectangle(img, new Point(matchLoc.x + imgCurrent.cols() + offsetX, matchLoc.y), new Point(matchLoc.x + imgCurrent.cols() + imgCurrent.cols() + offsetX,matchLoc.y + img1.rows()), COLOR_GREEN);
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 *
	 * @param event
	 *            the push button event
	 * @return 
	 */
	@FXML
	protected void start3dModelPreview(ActionEvent event)
	{		
		if(startPreview) {
			startPreview = false;
			testOpenGL();
		}		
	}
	
	@FXML
	protected void drawMatchesSwitch(ActionEvent event)
	{		
		drawMatches = ! drawMatches;
	}
	
	@FXML
	protected void drawProjectedCube(ActionEvent event)
	{		
		drawProjectedCube = ! drawProjectedCube;
	}
	
	@FXML
	protected void test(ActionEvent event)
	{		
		GL11.glReadBuffer(GL11.GL_FRONT);
		int width = Display.getDisplayMode().getWidth();
		int height= Display.getDisplayMode().getHeight();
		int bpp = 4; // Assuming a 32-bit display with a byte each for red, green, blue, and alpha.
		ByteBuffer buffer = BufferUtils.createByteBuffer(width * height * bpp);
		GL11.glReadPixels(0, 0, width, height, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, buffer );
		
		File file = new File("C:/Users/Marius/Desktop/test123.PNG"); // The file to save to.
		String format = "PNG"; // Example: "PNG" or "JPG"
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		   
		for(int x = 0; x < width; x++) 
		{
		    for(int y = 0; y < height; y++)
		    {
		        int i = (x + (width * y)) * bpp;
		        int r = buffer.get(i) & 0xFF;
		        int g = buffer.get(i + 1) & 0xFF;
		        int b = buffer.get(i + 2) & 0xFF;
		        image.setRGB(x, height - (y + 1), (0xFF << 24) | (r << 16) | (g << 8) | b);
		    }
		}
		   
		try {
		    ImageIO.write(image, format, file);
		} catch (IOException e) { e.printStackTrace(); }
	}
	
	private static boolean startPreview = true;
	
	/**
	 * The action triggered by pushing the button on the GUI
	 *
	 * @param event
	 *            the push button event
	 */
	@FXML
	protected void startCamera(ActionEvent event)
	{			
		if (!this.cameraActive)
		{
			// start the video capture
			this.capture.open(cameraId);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						//countFrames++;
						//System.out.println(countFrames);
						
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(currentFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.button.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Impossible to open the camera connection...");
			}
		}
		else
		{
			
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.button.setText("Start Camera");
			
			// stop the timer
			this.stopAcquisition();
			
			//testOpenGL();
		}
	}
	
	@SuppressWarnings("unused")
	private void chessBoardAR(Mat img) {
		Size chessSize = new Size(9,6);
		MatOfPoint2f corners = new MatOfPoint2f();
		boolean foundChessboard = Calib3d.findChessboardCorners(img, chessSize, corners);
		Calib3d.drawChessboardCorners(img, chessSize, corners, foundChessboard);
		
		float BoardBoxSize=3;//distance between 2 correns //change it accrounding to your pattern . megger it in cm or mm.
        //your unit of meggerment will consider as object point units.
		List<Point3> listCorners3d = new ArrayList<Point3>();
		for (int j = 0; j < chessSize.height; j++) {
			for (int i = 0; i < chessSize.width; i++) {
				listCorners3d.add(new Point3(i * BoardBoxSize, j * BoardBoxSize, 0));
			}
		}
		MatOfPoint3f corners3d = new MatOfPoint3f();
		corners3d.fromList(listCorners3d);
		
		// Camera internals
	    double focal_length = imgCurrent.cols(); // Approximate focal length.
	    Point center = new Point(imgCurrent.cols()/2,imgCurrent.rows()/2);
		
		Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
		int row = 0, col = 0;
		cameraMatrix.put(row ,col, focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1 );

		Mat rvec = new Mat();
		Mat tvec = new Mat();
		Calib3d.solvePnP(corners3d, corners, cameraMatrix, new MatOfDouble(), rvec, tvec);
	
		MatOfPoint3f objectPoints = new MatOfPoint3f();
		List<Point3> listObjectPoints = new ArrayList<Point3>();
	  //listObjectPoints.add(new Point3(  0,  0,  0));
		listObjectPoints.add(new Point3(  0,  0,-10));
		listObjectPoints.add(new Point3( 10,  0,  0));
		listObjectPoints.add(new Point3(  0, 10,  0));
		listObjectPoints.add(new Point3( 10, 10,  0));
		listObjectPoints.add(new Point3( 10, 10,-10));
		listObjectPoints.add(new Point3(  0, 10,-10));
		listObjectPoints.add(new Point3(  10, 0,-10));
		
		listObjectPoints.add(new Point3(33, 0,-1.0));//(x,y,z)
		listObjectPoints.add(new Point3(12, 8,-1.0));
		listObjectPoints.add(new Point3(20, 8,-1.0));
		listObjectPoints.add(new Point3(20, 0,-1.0));
		
		objectPoints.fromList(listObjectPoints);
		MatOfDouble distCoeffs = new MatOfDouble();
		MatOfPoint2f imagePoints = new MatOfPoint2f();
		Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

		List<Point> listCorners = corners.toList();
		List<Point> listPoints = imagePoints.toList();
		
		Core.line(img, listCorners.get(0), listPoints.get(0), new Scalar(0, 255, 0), 3);
		Core.line(img, listCorners.get(0), listPoints.get(1), new Scalar(255, 0, 0), 3);
		Core.line(img, listCorners.get(0), listPoints.get(2), new Scalar(0, 0, 255), 3);
		Core.line(img, listPoints.get(2), listPoints.get(3), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(3), listPoints.get(1), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(3), listPoints.get(4), new Scalar(0, 255, 255), 3);
		Core.line(img, listPoints.get(2), listPoints.get(5), new Scalar(0, 255, 255), 3);
		Core.line(img, listPoints.get(1), listPoints.get(6), new Scalar(0, 255, 255), 3);	

		Core.line(img, listPoints.get(0), listPoints.get(5), new Scalar(0, 255, 255), 3);
		Core.line(img, listPoints.get(0), listPoints.get(6), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(4), listPoints.get(5), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(4), listPoints.get(6), new Scalar(0, 255, 255), 3);	

	/*
		Core.line(img, listCorners.get(3), listPoints.get(3), COLOR_RED, 3);
		Core.line(img, listCorners.get(21), listPoints.get(4), COLOR_RED, 3);
		Core.line(img, listCorners.get(23), listPoints.get(5), COLOR_RED, 3);
		Core.line(img, listCorners.get(5), listPoints.get(6), COLOR_RED, 3);	

		Core.line(img, listCorners.get(3), listCorners.get(5), COLOR_BLUE, 3);
		Core.line(img, listCorners.get(5), listCorners.get(23), COLOR_BLUE, 3);
		Core.line(img, listCorners.get(23), listCorners.get(21), COLOR_BLUE, 3);
		Core.line(img, listCorners.get(21), listCorners.get(3), COLOR_BLUE, 3);

		Core.line(img, listPoints.get(3), listPoints.get(4), COLOR_GREEN, 3);
		Core.line(img, listPoints.get(4), listPoints.get(5), COLOR_GREEN, 3);
		Core.line(img, listPoints.get(5), listPoints.get(6), COLOR_GREEN, 3);
		Core.line(img, listPoints.get(3), listPoints.get(6), COLOR_GREEN, 3);		
	*/	
/*		
 		//System.out.println("Rotation:\n");
		//System.out.println(rvec.dump());		
		Size size = rvec.size();
		for (int i = 0; i < size.height; i++) {
		    for (int j = 0; j < size.width; j++) { // j always 0: size.width == 0
		        double[] data = rvec.get(i, j);
		        for(double x : data) {
		        	double degree = 180 * x / Math.PI;
		        	//if( degree < 0 ) degree += 360.0;
		        	switch(i) {
	        		case 0:
	        			rotX = (int) Math.round(degree);
	        			System.out.println("X: " + Math.round(degree));
	        			break;
	        		case 1:
	        			rotY = (int) Math.round(degree);
	        			System.out.println("Y: " + Math.round(degree));
	        			break;
	        		case 2:
	        			rotZ = (int) Math.round(degree);
	        			System.out.println("Z: " + rotZ);
	        			break;
	        		default:	        	
		        	}
		        	//System.out.println(Math.round(degree));
		        }
		    }
		}
		
		//Calib3d.solvePnP(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
		//Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
		//Calib3d.solvePnPRansac(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
		//Calib3d.solvePnP(corners3d, corners, cameraMatrix, new MatOfDouble(), rvec, tvec);
		Calib3d.solvePnPRansac(corners3d, corners, cameraMatrix, new MatOfDouble(), rvec, tvec);
	
		Mat dst = new Mat();
		Calib3d.Rodrigues(rvec, dst);
		
		//dst = homography;
		//System.out.println("???\n" + dst.dump());
		
		Mat projMatrix = new Mat(3,4,CvType.CV_64FC1);	
		projMatrix.put(0, 0, dst.get(0, 0));
		projMatrix.put(0, 1, dst.get(0, 1));
		projMatrix.put(0, 2, dst.get(0, 2));
		projMatrix.put(0, 3, 0);
		projMatrix.put(1, 0, dst.get(1, 0));
		projMatrix.put(1, 1, dst.get(1, 1));
		projMatrix.put(1, 2, dst.get(1, 2));
		projMatrix.put(1, 3, 0);
		projMatrix.put(2, 0, dst.get(2, 0));
		projMatrix.put(2, 1, dst.get(2, 1));
		projMatrix.put(2, 2, dst.get(2, 2));
		projMatrix.put(2, 3, 0);
		
		//out("homography: \n" + homography.dump()); 
		//System.out.println();
		//out("projmatrix: \n" + projMatrix.dump()); 
	
		
		Mat rotMatrix = new Mat();
		Mat transVect = new Mat();
		Mat rotMatrixX = new Mat();
		Mat rotMatrixY = new Mat();
		Mat rotMatrixZ = new Mat();
		
		Mat eulerAngles = new Mat();		
	
		//Calib3d.decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect);
		Calib3d.decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles);
	
		
		System.out.println(eulerAngles.dump());
  		System.out.println("--------------\n");
*/
	}
	
	private void matchTemplate(Mat img) {
		int match_method = Imgproc.TM_SQDIFF; //Imgproc.TM_CCOEFF;// 
		
		// / Create the result matrix
	    int result_cols = img.cols() - img1.cols() + 1;
	    int result_rows = img.rows() - img1.rows() + 1;
	    Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);

	    // Do the Matching and Normalize
	    Imgproc.matchTemplate(img, img1, result, match_method);
	    //Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());

	    // Localizing the best match with minMaxLoc
	    MinMaxLocResult mmr = Core.minMaxLoc(result);
	    
	    //System.out.println(mmr.minVal + "\n" + mmr.maxVal + "\n\n");
	    //System.out.println(Math.round(100.0 * mmr.minVal));
	    double min1 = mmr.minVal;
	    
	    int result_cols2 = img.cols() - img2.cols() + 1;
	    int result_rows2 = img.rows() - img2.rows() + 1;
	    Mat result2 = new Mat(result_rows2, result_cols2, CvType.CV_32FC1);
	    Imgproc.matchTemplate(img, img2, result2, match_method);
	    //Core.normalize(result2, result2, 0, 1, Core.NORM_MINMAX, -1, new Mat());
	    
	    MinMaxLocResult mmr2 = Core.minMaxLoc(result2);
	    
	    double min2 = mmr2.minVal;
	    
	    int result_cols3 = img.cols() - img3.cols() + 1;
	    int result_rows3 = img.rows() - img3.rows() + 1;
	    Mat result3 = new Mat(result_rows3, result_cols3, CvType.CV_32FC1);
	    Imgproc.matchTemplate(img, img3, result3, match_method);
	    //Core.normalize(result2, result2, 0, 1, Core.NORM_MINMAX, -1, new Mat());
	    
	    MinMaxLocResult mmr3 = Core.minMaxLoc(result3);
	    
	    double min3 = mmr3.minVal;
/*
	    System.out.println();
	    System.out.println("img1: " + min1);					    
	    System.out.println("img2: " + min2);					    
	    System.out.println("img3: " + min3);
*/
	    //if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
	    //    matchLoc = mmr.minLoc;
	    //} else {
	    //    matchLoc = mmr.maxLoc;
	    //}				    

	    // / Show me what you got
	    if (min1 < min2 && min1 < min3 ) {
	    	//true) {
			//System.out.println("red");
			textImageName = " (Bild 1)";
			matchLoc = mmr.minLoc;
			imgCurrent = img1;
	    	//Imgproc.rectangle(img, matchLoc, new Point(matchLoc.x + img1.cols(),matchLoc.y + img1.rows()), new Scalar(0, 255, 0));
		} else if (min2 < min3 && min2 < min1) {
			//System.out.println("blue");
			textImageName = " (Bild 2)";
			matchLoc = mmr2.minLoc;
			imgCurrent = img2;
			//Imgproc.rectangle(img, matchLoc, new Point(matchLoc.x + img2.cols(),matchLoc.y + img2.rows()), new Scalar(0, 255, 0));
		}	else if (min3 < min1 && min3 < min2) {
			//System.out.println("green");
			textImageName = " (Bild 3)";
			matchLoc = mmr3.minLoc;
			imgCurrent = img3;
			//Imgproc.rectangle(img, matchLoc, new Point(matchLoc.x + img3.cols(),matchLoc.y + img3.rows()), new Scalar(0, 255, 0));
		}	    
	}
	
	private MatOfPoint3f objectPoints3d;
	private MatOfPoint2f scenePoints2d;
	
	private Mat homography;
	
	private void matchFeaturePoints(Mat img, MatOfPoint3f objectPoints3d, MatOfPoint2f scenePoints2d) {
		
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.ORB);	
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);

		// get SOURCE image keypoints from source image
		MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();			
        featureDetector.detect(imgCurrent, objectKeyPoints);
        
        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(imgCurrent, objectKeyPoints, objectDescriptors);
        
        //printKeyPoints(objectKeyPoints); 
        //Features2d.drawKeypoints(imgCurrent, objectKeyPoints, img, COLOR_GREEN, 0);
        
        // get SCENE image keypoints
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        featureDetector.detect(img, sceneKeyPoints);

        // Computing descriptors in background image
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(img, sceneKeyPoints, sceneDescriptors); 
        
        //printKeyPoints(sceneKeyPoints);        
        //Features2d.drawKeypoints(img, sceneKeyPoints, img, COLOR_GREEN, 0);
        
        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        
        objectDescriptors.convertTo(objectDescriptors, CvType.CV_32F);
        sceneDescriptors.convertTo(sceneDescriptors, CvType.CV_32F);
        
        Mat descriptorsLarge = new Mat();
        Mat descriptorsSmall = new Mat();

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        extractor.compute(imgCurrent, objectKeyPoints, descriptorsLarge);
        extractor.compute(img, sceneKeyPoints, descriptorsSmall);
        MatOfDMatch matches2 = new MatOfDMatch();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        matcher.clear();
        matcher.match(descriptorsLarge, descriptorsSmall, matches2);
        //Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, matches2, img);
        
        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2); // matching object and scene images   
        //Features2d.drawMatchesKnn(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, matches2, img);

        //System.out.println("len object : " + objectKeyPoints.size());
        //System.out.println("len scene  : " + sceneKeyPoints.size());
        //System.out.println("len matches: " + matches.size());
        //int count = 0;
        //for(MatOfDMatch match : matches) {
        	//double x, y;
        	//x = match[1];
        	//System.out.println(match.dump());   
            //count++;
        //}
        //System.out.println("Number of matches: " + count);
        //System.out.println("--------------------------");         
        
        //Features2d.drawMatches2(imgCurrent, objectKeyPoints, img, sceneKeyPoints, matches, output);
        //img = output;
        
        //Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, matches2, img);
        //Features2d.drawMatchesKnn(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, matches, img);
        //Features2d.drawMatchesKnn(imgCurrent, objectKeyPoints, img, sceneKeyPoints, matches, img, newKeypointColor, newKeypointColor, null, 2);

        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();
        List<MatOfDMatch> list = new LinkedList<MatOfDMatch>();

        //nndrRatio = 1;

        //count = 0;
        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);
                list.add(matofDMatch);
                //count++;
            }
        }
        //System.out.println("Number of good matches: " + count);
        //System.out.println("--------------------------");

        MatOfDMatch goodMatches = new MatOfDMatch();
        if (//true) {
        	goodMatchesList.size() >= 10) {
            //System.out.println("Object Found!!!");
	    	textImage = "Objekt erkannt!";
	    	textImage += textImageName;

            List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
            List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

            LinkedList<Point> objectPoints = new LinkedList<>();
            LinkedList<Point> scenePoints = new LinkedList<>();            

            //System.out.println("good matches size: " + goodMatchesList.size());
            //System.out.println();            

            Point3 arrayPoints3D[] = new Point3[goodMatchesList.size()];
            Point arrayPoints2D[] = new Point[goodMatchesList.size()];
            
            MatOfPoint3f objPoints = new MatOfPoint3f();
            MatOfPoint2f imgPoints = new MatOfPoint2f();            

            for (int i = 0; i < goodMatchesList.size(); i++) {
            	Point oPoint = objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt;
                objectPoints.addLast(oPoint);
                //System.out.println(oPoint.toString());
            	Point sPoint = scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt;
                //System.out.println(sPoint.toString());
                scenePoints.addLast(sPoint);
                //System.out.println();
                
                arrayPoints3D[i]= new Point3(sPoint.x, sPoint.y, 0);
                arrayPoints2D[i] = new Point(oPoint.x, oPoint.y);
            }

            objPoints.fromArray(arrayPoints3D);
            imgPoints.fromArray(arrayPoints2D);
            
            List<Point3> list3d = objPoints.toList();
            List<Point> list2d = imgPoints.toList();
            if(list3d.size() == list2d.size()) {
            	for(int j = 0; j < list3d.size(); j++) {
            		//System.out.println(list3d.get(j).toString());
            		//System.out.println(list2d.get(j).toString());
            		//System.out.println();
            	}
            }
            
            objectPoints3d = objPoints;
            scenePoints2d = imgPoints;

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);

            homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 5);
            //out("homography: \n" + homography.dump());     
  //*              		

//*/
            testPoseEstimation(objPoints, imgPoints);

            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            obj_corners.put(0, 0, new double[]{0, 0});
            obj_corners.put(1, 0, new double[]{imgCurrent.cols(), 0});
            obj_corners.put(2, 0, new double[]{imgCurrent.cols(), imgCurrent.rows()});
            obj_corners.put(3, 0, new double[]{0, imgCurrent.rows()});

            //System.out.println("Transforming object corners to scene corners...");
            Core.perspectiveTransform(obj_corners, scene_corners, homography);
            
            Point3 leftUp3d =    new Point3(0,0,0);
            Point3 rightUp3d =   new Point3(1,0,0);
            Point3 rightDown3d = new Point3(1,1,0);
            Point3 leftDown3d =  new Point3(0,1,0);
            
            Point leftUp = new Point(scene_corners.get(0, 0));
            Point rightUp = new Point(scene_corners.get(1, 0));
            Point rightDown = new Point(scene_corners.get(2, 0));
            Point leftDown = new Point(scene_corners.get(3, 0));
            
            Core.line(img, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), COLOR_GREEN, 4);
            Core.line(img, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), COLOR_GREEN, 4);
            Core.line(img, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), COLOR_GREEN, 4);
            Core.line(img, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), COLOR_GREEN, 4);
            
    		Core.circle(img, leftUp, 10, COLOR_RED, 3);
    		Core.circle(img, rightUp, 10, COLOR_RED, 3);
    		Core.circle(img, leftDown, 10, COLOR_RED, 3);
    		Core.circle(img, rightDown, 10, COLOR_RED, 3);
            
            System.out.println("leftUp   : " + leftUp.toString());
            System.out.println("rightUp  : " + rightUp.toString());
            System.out.println("leftDown : " + leftDown.toString());
            System.out.println("rightDown: " + rightDown.toString());
            
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
    		cameraMatrix.put(0 ,0, imgCurrent.cols(), 0, imgCurrent.cols()/2, 0, imgCurrent.cols(), imgCurrent.rows()/2, 0, 0, 1 );
    		
    		MatOfPoint3f objPointsTest = new MatOfPoint3f(leftUp3d, rightUp3d, rightDown3d, leftDown3d);
            MatOfPoint2f imgPointsTest = new MatOfPoint2f(leftUp, rightUp, rightDown, leftDown); 
    		
    		Mat rvec = new Mat();
    		Mat tvec = new Mat();
    		Calib3d.solvePnP(objPointsTest, imgPointsTest, cameraMatrix, new MatOfDouble(), rvec, tvec);
    		
    		if(drawProjectedCube) {
        		
        		MatOfPoint3f objectPointsCube = new MatOfPoint3f();
        		List<Point3> listObjectPoints = new ArrayList<Point3>();
        		double height = 0.25;
        	  //listObjectPoints.add(new Point3(  0, 0, 0)); // origin
        		listObjectPoints.add(new Point3(  0, 0,height));
        		listObjectPoints.add(new Point3(  1, 0, 0));
        		listObjectPoints.add(new Point3(  0, 1, 0));
        		listObjectPoints.add(new Point3(  1, 1, 0));
        		listObjectPoints.add(new Point3(  1, 1,height));
        		listObjectPoints.add(new Point3(  0, 1,height));
        		listObjectPoints.add(new Point3(  1, 0,height));
        	/*	
        		listObjectPoints.add(new Point3(33, 0,-1.0));//(x,y,z)
        		listObjectPoints.add(new Point3(12, 8,-1.0));
        		listObjectPoints.add(new Point3(20, 8,-1.0));
        		listObjectPoints.add(new Point3(20, 0,-1.0));
        	*/	
        		objectPointsCube.fromList(listObjectPoints);
        		MatOfDouble distCoeffs = new MatOfDouble();
        		MatOfPoint2f imagePoints = new MatOfPoint2f();
        		Calib3d.projectPoints(objectPointsCube, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

        		Point origin = leftUp;
        		List<Point> listPoints = imagePoints.toList();
        		Core.line(img, origin, listPoints.get(0), new Scalar(0, 255, 255), 3);
        		Core.line(img, origin, listPoints.get(1), new Scalar(0, 255, 255), 3);
        		Core.line(img, origin, listPoints.get(2), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(2), listPoints.get(3), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(3), listPoints.get(1), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(3), listPoints.get(4), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(2), listPoints.get(5), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(1), listPoints.get(6), new Scalar(0, 255, 255), 3);	

        		Core.line(img, listPoints.get(0), listPoints.get(5), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(0), listPoints.get(6), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(4), listPoints.get(5), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(4), listPoints.get(6), new Scalar(0, 255, 255), 3);    			
    		}	
    		
    		//MatOfPoint2f imagePoints = new MatOfPoint2f();
    		//Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, new MatOfDouble(), imagePoints);
            
    		Size sizeTrans = tvec.size();
    		for (int i = 0; i < sizeTrans.height; i++) {
    		    for (int j = 0; j < sizeTrans.width; j++) { // j always 0: size.width == 0
    		        double[] data = tvec.get(i, j);
    		        for(double x : data) {
    		        	switch(i) {
    	        		case 0:
    	        			transX = (int) (10*x) -3;
    	        			PoseEstimationValues.posX = transX;
    	        			PoseEstimationValues.posX /= 5;
    	        			System.out.println("X: " + transX);
    	        			break;
    	        		case 1:
    	        			transY = (int) (10*x);
    	        			PoseEstimationValues.posY = transY;
    	        			PoseEstimationValues.posY /= 5;
    	        			System.out.println("Y: " + transY);
    	        			break;
    	        		case 2:
    	        			transZ = (int) (10*x)-10;
    	        			PoseEstimationValues.posZ = transZ;
    	        			PoseEstimationValues.posZ /= 5;
    	        			System.out.println("Z: " + transZ);
    	        			break;
    	        		default:	        	
    		        	}
    		        }
    		    }
    		}
            System.out.println("\nRotation:");
            Size size = rvec.size();
    		for (int i = 0; i < size.height; i++) {
    		    for (int j = 0; j < size.width; j++) { // j always 0: size.width == 0
    		        double[] data = rvec.get(i, j);
    		        for(double x : data) {
    		        	double degree = 180 * x / Math.PI;
    		        	//if( degree < 0 ) degree += 360.0;
    		        	switch(i) {
    	        		case 0:
    	        			rotX = (int) Math.round(degree);
    	        			PoseEstimationValues.rotX = rotX;
    	        			System.out.println("X: " + Math.round(degree));
    	        			break;
    	        		case 1:
    	        			rotY = (int) Math.round(degree);
    	        			PoseEstimationValues.rotY = rotY;
    	        			System.out.println("Y: " + Math.round(degree));
    	        			break;
    	        		case 2:
    	        			rotZ = (int) Math.round(degree);
    	        			PoseEstimationValues.rotZ = -rotZ;
    	        			System.out.println("Z: " + rotZ);
    	        			break;
    	        		default:	        	
    		        	}
    		        	//System.out.println(Math.round(degree));
    		        }
    		    }
    		}
            
            System.out.println("-------");

            //System.out.println("Drawing matches image...");
            //MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);
            
            /*
            for(unsigned int match_index = 0; match_index < good_matches.size(); ++match_index)
			{
			    cv::Point3f point3d_model = list_points3d_model[ good_matches[match_index].trainIdx ];   // 3D point from model
			    cv::Point2f point2d_scene = keypoints_scene[ good_matches[match_index].queryIdx ].pt;    // 2D point from the scene
			    list_points3d_model_match.push_back(point3d_model);                                      // add 3D point
			    list_points2d_scene_match.push_back(point2d_scene);                                      // add 2D point
			}
             */
            
            //Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, goodMatches, new Mat(), newKeypointColor, newKeypointColor, new MatOfByte(), 2);
            //Features2d.drawMatchesKnn(imgCurrent, objectKeyPoints, output, sceneKeyPoints, list, img);
            
            //Highgui.imwrite("output//outputImage.jpg", outputImage);
            //Highgui.imwrite("output//matchoutput.jpg", matchoutput);
            //Highgui.imwrite("output//img.jpg", img);
        } else {
            //System.out.println("----- Object Not Found -----");

	    	textImage = "Bitte Objekt fixieren...";
	    	
	    	imgCurrent = img3;
        }
        
        if(drawMatches) {
        	//Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, matches2, img);
        	Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, goodMatches, img);
        }
        //Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, goodMatches, img);//DRAW!dfd
        //Features2d.drawMatchesKnn(imgCurrent, objectKeyPoints, output, sceneKeyPoints, list, img);    
        //Features2d.drawMatchesKnn(imgCurrent, objectKeyPoints, output, sceneKeyPoints, goodMatchesList, img);
        
        
        //Imgproc.rectangle(img, new Point(10,20), new Point(100,50), new Scalar(0, 255, 0));
	}
	
	private void testPoseEstimation(MatOfPoint3f objPoints, MatOfPoint2f imgPoints) {

		tCount++;
		if(tCount < 7) {
			return;
		}
		tCount = 0;
/*        
		float offsetX = 0;//112f;
		float offsetY = 0;//112f;
		
		MatOfPoint3f objectPoints = new MatOfPoint3f(new Point3(-1,-1, 0), 
													 new Point3( 1,-1, 0), 
													 new Point3(-1, 1, 0), 
													 new Point3( 1, 1, 0));
		
		MatOfPoint2f imagePoints = new MatOfPoint2f( new Point (-1+offsetX,-1+offsetY), 
													 new Point ( 1+offsetX,-0.5f+offsetY), 
													 new Point (-1+offsetX, 1+offsetY), 
													 new Point ( 1+offsetX, 0.5f+offsetY));
*/		
		// Camera internals
	    double focal_length = imgCurrent.cols(); // Approximate focal length.
	    Point center = new Point(imgCurrent.cols()/2,imgCurrent.rows()/2);
		
		Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
		int row = 0, col = 0;
		cameraMatrix.put(row ,col, focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1 );
	
		
		/*
		 * 		    
		 *  double x = Math.random() * 100 - 50;
		    double y = Math.random() * 100 - 50;
		    Mat point2D = new Mat(2,1,CvType.CV_64FC1);
		    point2D.put(0, 0, x, y);
		    Mat point3D = new Mat(3,1,CvType.CV_64FC1);
		    point3D.put(0, 0, 0, y, x);
		    
		    points2d.push_back(point2D);
		    points3d.push_back(point3D);
		 */
		
		//System.out.println(objectPoints.dump());
		//System.out.println(imagePoints.dump());
		
		Mat rvec = new Mat();
		Mat tvec = new Mat();
		
		@SuppressWarnings("unused")
		MatOfDouble mRMat = new MatOfDouble(3, 3, CvType.CV_32F);
		
		if (objPoints != null && imgPoints != null) {
			//Calib3d.solvePnP(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
			Calib3d.solvePnPRansac(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
			//Calib3d.solvePnP(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
			//Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
			//Calib3d.solvePnPRansac(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
		} else {
			return;
		}
		
		Size size = rvec.size();
		for (int i = 0; i < size.height; i++) {
		    for (int j = 0; j < size.width; j++) { // j always 0: size.width == 0
		        double[] data = rvec.get(i, j);
		        for(double x : data) {
		        	double degree = 180 * x / Math.PI;
		        	//if( degree < 0 ) degree += 360.0;
		        	switch(i) {
	        		case 0:
	        			rotX = (int) Math.round(degree);
	        			System.out.println("X: " + Math.round(degree));
	        			break;
	        		case 1:
	        			rotY = (int) Math.round(degree);
	        			System.out.println("Y: " + Math.round(degree));
	        			break;
	        		case 2:
	        			rotZ = (int) Math.round(degree);
	        			System.out.println("Z: " + rotZ);
	        			break;
	        		default:	        	
		        	}
		        	//System.out.println(Math.round(degree));
		        }
		    }
		}
		
		Mat dst = new Mat();
		Calib3d.Rodrigues(rvec, dst);
		
		//dst = homography;
		//System.out.println("???\n" + dst.dump());
		
		Mat projMatrix = new Mat(3,4,CvType.CV_64FC1);	
		projMatrix.put(0, 0, dst.get(0, 0));
		projMatrix.put(0, 1, dst.get(0, 1));
		projMatrix.put(0, 2, dst.get(0, 2));
		projMatrix.put(0, 3, 0);
		projMatrix.put(1, 0, dst.get(1, 0));
		projMatrix.put(1, 1, dst.get(1, 1));
		projMatrix.put(1, 2, dst.get(1, 2));
		projMatrix.put(1, 3, 0);
		projMatrix.put(2, 0, dst.get(2, 0));
		projMatrix.put(2, 1, dst.get(2, 1));
		projMatrix.put(2, 2, dst.get(2, 2));
		projMatrix.put(2, 3, 0);
		
		//out("homography: \n" + homography.dump()); 
		//System.out.println();
		//out("projmatrix: \n" + projMatrix.dump()); 

		
		Mat rotMatrix = new Mat();
		Mat transVect = new Mat();
		Mat rotMatrixX = new Mat();
		Mat rotMatrixY = new Mat();
		Mat rotMatrixZ = new Mat();
		
		Mat eulerAngles = new Mat();		

		//Calib3d.decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect);
		Calib3d.decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles);

		
		System.out.println(eulerAngles.dump());
		
/*		
		projMatrix.put(0, 0, rVecArray[0]);
	    projMatrix.put(0, 1, rVecArray[1]);
	    projMatrix.put(0, 2, rVecArray[2]);
	    projMatrix.put(0, 3, 0);
	    projMatrix.put(1, 0, rVecArray[3]);
	    projMatrix.put(1, 1, rVecArray[4]);
	    projMatrix.put(1, 2, rVecArray[5]);
	    projMatrix.put(1, 3, 0);
	    projMatrix.put(2, 0, rVecArray[6]);
	    projMatrix.put(2, 1, rVecArray[7]);
	    projMatrix.put(2, 2, rVecArray[8]);
	    projMatrix.put(2, 3, 0);

		Calib3d.decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect);
		//Calib3d.decomposeProjectionMatrix(Mat projMatrix, cameraMatrix, Mat rotMatrix, Mat transVect, Mat rotMatrixX, Mat rotMatrixY, Mat rotMatrixZ, Mat eulerAngles);
*/	
		//double rotX = rvec.get(0, 0);
//		System.out.println("Rotation:\n");
		//System.out.println(rvec.dump());
//		System.out.println(rotMatrix.dump());
		/*
		Size size = rvec.size();
		for (int i = 0; i < size.height; i++) {
		    for (int j = 0; j < size.width; j++) { // j always 0: size.width == 0
		        double[] data = rvec.get(i, j);
		        for(double x : data) {
		        	double degree = 180 * x / Math.PI;
		        	//if( degree < 0 ) degree += 360.0;
		        	switch(i) {
	        		case 0:
	        			rotX = (int) Math.round(degree);
	        			System.out.println("X: " + Math.round(degree));
	        			break;
	        		case 1:
	        			rotY = (int) Math.round(degree);
	        			System.out.println("Y: " + Math.round(degree));
	        			break;
	        		case 2:
	        			rotZ = (int) Math.round(degree);
	        			System.out.println("Z: " + rotZ);
	        			break;
	        		default:	        	
		        	}
		        	//System.out.println(Math.round(degree));
		        }
		    }
		}
		

		System.out.println("Translation:");
		Size sizeTrans = rvec.size();
		for (int i = 0; i < sizeTrans.height; i++) {
		    for (int j = 0; j < sizeTrans.width; j++) { // j always 0: size.width == 0
		        double[] data = tvec.get(i, j);
		        for(double x : data) {
		        	//double degree = 180 * x / Math.PI;
		        	//if( degree < 0 ) degree += 360.0;
		        	switch(i) {
	        		case 0:
	        			transX = -((int)Math.round(x)+128) - 200;
	        			System.out.println("X: " + transX);
	        			break;
	        		case 1:
	        			transY = -((int)Math.round(x)+128) - 100;
	        			System.out.println("Y: " + transY);
	        			break;
	        		case 2:
	        			transZ = (int)Math.round(x) -255;
	        			System.out.println("Z: " + transZ);
	        			break;
	        		default:	        	
		        	}
		        }
		    }
		}
		//*/
		out("----------");
		
		//*
		//System.out.println("Rotation: \n\n" + rvec.dump());
		//System.out.println();
		//System.out.println("Translation: \n\n" + tvec.dump());
		//*/
	}
	
	private int tCount = 0;
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
	public static void printKeyPoints(MatOfKeyPoint keyPoints) {
        KeyPoint[] keypoints = keyPoints.toArray();
        
        int count = 0;
        for(KeyPoint keypoint : keypoints) {
            System.out.println(keypoint.toString());   
            count++;
        }
        System.out.println("Number of keypoints: " + count); 
	}
	
	private static boolean printOut = true;	
	private static void out(String text) {
		if(printOut) {
			System.out.println(text);
		}
	}
	
}
