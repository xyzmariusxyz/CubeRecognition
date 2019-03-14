package main;

import org.opencv.core.Core;

//import java.io.File;
//import java.lang.reflect.Field;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.fxml.FXMLLoader;

/**
 * The main class for a JavaFX application. It creates and handle the main
 * window with its resources (style, graphics, etc.).
 * 
 */
public class MainCubeRecognition extends Application
{
	
	@Override
	public void start(Stage primaryStage)
	{
		try
		{
			// load the FXML resource
			FXMLLoader loader = new FXMLLoader(getClass().getResource("MainCubeRecognition.fxml"));
			// store the root element so that the controllers can use it
			BorderPane rootElement = (BorderPane) loader.load();
			// create and style a scene
			Scene scene = new Scene(rootElement, 900, 600);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			// create the stage with the given title and the previously created
			// scene
			primaryStage.setTitle("Object Recognition Based On Digital Twin");
			primaryStage.setScene(scene);
			// show the GUI
			primaryStage.show();
			
			// set the proper behavior on closing the application
			ImageRecognition controller = loader.getController();
			controller.startCamera(null);
			primaryStage.setOnCloseRequest((new EventHandler<WindowEvent>() {
				public void handle(WindowEvent we)
				{
					controller.setClosed();
				}
			}));
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	/**
	 * For launching the application...
	 * 
	 * @param args
	 *            optional params
	 * @throws SecurityException 
	 * @throws NoSuchFieldException 
	 * @throws IllegalAccessException 
	 * @throws IllegalArgumentException 
	 */
	public static void main(String[] args) throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException
	{		
		/*
		String libraryPath = "C:\\OpenCV-2.4.6\\build\\java\\x64";//"OpenCV_lib"; //C:\OpenCV-2.4.6\build\java\x64
		System.setProperty("java.library.path", libraryPath); 
		Field sysPath = ClassLoader.class.getDeclaredField("sys_paths");
		sysPath.setAccessible(true); 
		sysPath.set(null, null);
		/*	
		//all opencv libs must be copyed to OpenCV_lib in the project workspace
        File folder = new File("OpenCV_lib/");
        File[] listOfFiles = folder.listFiles(); 

	    for (int i = 0; i < listOfFiles.length; i++) {
	        if (listOfFiles[i].isFile() && listOfFiles[i].getName().endsWith(".dll")) {
	            File lib = new File("OpenCV_lib/" + listOfFiles[i].getName()); 
	            System.load(lib.getAbsoluteFile().toString());
	        }
	    }
	    
	    //*/   
		// load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//ImageRecognition.testOpenGL();
		launch(args);
	}
}
