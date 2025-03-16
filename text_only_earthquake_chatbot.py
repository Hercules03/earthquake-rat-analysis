"""
Text-Only Earthquake RAT Chatbot.
This version avoids PDF processing issues by using only text files.
"""

import os
import logging
from typing import List, Optional
from raglight.rat.simple_rat_api import RATPipeline
from raglight.models.data_source_model import FolderSource
from raglight.config.settings import Settings

class TextOnlyEarthquakeChatbot:
    """
    A RAT-based chatbot for earthquake analysis that uses text files only.
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "./text_knowledge_base",
        model_name: str = "llama3.2:3b-instruct-q4_K_M",
        reasoning_model_name: str = "deepseek-r1:7b",  # Updated to match your available model
        reflection: int = 2,
        vector_store_path: str = "./earthquake_vector_store"
    ):
        """
        Initialize the Text-Only Earthquake Chatbot.
        """
        # Setup logging
        Settings.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Set vector store path
        os.environ["CHROMA_PATH"] = vector_store_path
        
        # Create knowledge base directory if it doesn't exist
        os.makedirs(knowledge_base_path, exist_ok=True)
        
        # Create initial text files if the directory is empty
        self._create_sample_text_files(knowledge_base_path)
        
        # Prepare knowledge sources (only local folder, no GitHub repos)
        self.knowledge_sources = [FolderSource(path=knowledge_base_path)]
        
        # Initialize the RAT pipeline with only supported parameters
        self.pipeline = RATPipeline(
            knowledge_base=self.knowledge_sources,
            model_name=model_name,
            reasoning_model_name=reasoning_model_name,
            reflection=reflection
        )
        
        self.is_built = False
    
    def _create_sample_text_files(self, knowledge_base_path: str):
        """
        Create sample text files about earthquakes if they don't exist.
        """
        if not os.listdir(knowledge_base_path):
            self.logger.info("Creating sample text files about earthquakes...")
            
            # Japan Earthquakes
            with open(os.path.join(knowledge_base_path, "japan_earthquakes.txt"), "w") as f:
                f.write("""
                Japan Earthquakes: Facts and Information
                
                Japan experiences about 1,500 earthquakes every year. The country lies in a zone where several tectonic plates meet, making it one of the most seismically active areas in the world.
                
                Major tectonic plates affecting Japan:
                - Pacific Plate
                - Philippine Sea Plate
                - Eurasian Plate
                - North American Plate
                
                The movement and collision of these plates lead to frequent earthquakes. The Japan Trench, where the Pacific Plate subducts beneath the North American Plate, is particularly active.
                
                Notable earthquakes in Japan's history:
                
                1. Great Kanto Earthquake (1923)
                   - Magnitude: 7.9
                   - Death toll: ~140,000
                   - Devastated Tokyo and Yokohama
                
                2. Great Hanshin-Awaji Earthquake (1995)
                   - Magnitude: 6.9
                   - Death toll: ~6,400
                   - Severely damaged Kobe
                
                3. Tohoku Earthquake and Tsunami (2011)
                   - Magnitude: 9.0-9.1
                   - Death toll: ~19,500
                   - Triggered tsunami and Fukushima nuclear disaster
                   - One of the most powerful earthquakes ever recorded
                
                Japan's earthquake preparedness:
                - Advanced early warning systems
                - Strict building codes
                - Regular evacuation drills
                - Public education and awareness
                """)
            
            # Earthquake Prediction
            with open(os.path.join(knowledge_base_path, "earthquake_prediction.txt"), "w") as f:
                f.write("""
                Earthquake Prediction: Methods and Challenges
                
                Earthquake prediction remains one of the greatest challenges in seismology. Unlike weather forecasting, which has become increasingly accurate, earthquake prediction is still largely elusive.
                
                Current approaches to earthquake prediction:
                
                1. Statistical methods
                   - Analysis of historical earthquake patterns
                   - Identification of seismic gaps
                   - Recurrence intervals for fault segments
                
                2. Precursor monitoring
                   - Changes in ground water levels
                   - Radon gas emissions
                   - Electromagnetic field changes
                   - Animal behavior (controversial)
                
                3. Machine Learning approaches
                   - Neural networks for pattern recognition
                   - Feature extraction from seismic data
                   - Integration of multiple data sources
                   - Time series analysis of seismic activity
                
                Challenges in earthquake prediction:
                
                1. Complexity of fault systems
                   - Non-linear behavior
                   - Multiple interacting faults
                   - Three-dimensional complexity
                
                2. Limited data
                   - Relatively short historical record
                   - Sparse instrumentation in many regions
                   - Difficulties in measuring deep Earth processes
                
                3. False positives and negatives
                   - Social and economic costs of false alarms
                   - Dangerous consequences of missed predictions
                
                Machine Learning in earthquake prediction:
                
                Machine learning models have shown promise in recognizing patterns in seismic data that may precede earthquakes. Some approaches include:
                
                - Convolutional Neural Networks (CNNs) for analyzing seismograms
                - Recurrent Neural Networks (RNNs) for time series analysis
                - Random Forests for feature importance identification
                - Support Vector Machines for classifying precursor events
                
                Researchers continue to improve these models by incorporating more diverse data sources, including:
                
                - Satellite geodesy (GPS measurements of ground deformation)
                - InSAR (Interferometric Synthetic Aperture Radar)
                - Groundwater monitoring
                - Strain measurements
                - Electrical resistivity surveys
                """)
            
            # Tsunami Warning Systems
            with open(os.path.join(knowledge_base_path, "tsunami_warning.txt"), "w") as f:
                f.write("""
                Tsunami Warning Systems in Japan
                
                Japan has one of the most advanced tsunami warning systems in the world, developed in response to the country's long history of destructive tsunamis.
                
                Components of Japan's tsunami warning system:
                
                1. Seismic monitoring
                   - Dense network of seismometers throughout Japan
                   - Ocean bottom seismometers
                   - Real-time data processing and analysis
                
                2. Sea level monitoring
                   - Tide gauges along the coast
                   - Offshore buoys with pressure sensors
                   - GPS-equipped buoys (DART system)
                
                3. Warning dissemination
                   - J-Alert nationwide warning system
                   - Automatic broadcasting on TV and radio
                   - Mobile phone alerts
                   - Coastal sirens and loudspeakers
                
                4. Evacuation infrastructure
                   - Clearly marked evacuation routes
                   - Tsunami shelters and evacuation buildings
                   - Seawalls and breakwaters
                
                Warning timeline:
                
                - Preliminary tsunami warning: Issued within 2-3 minutes after earthquake detection
                - Updated warning: Issued within 10 minutes with more accurate tsunami height predictions
                - Final warning: Continuous updates as tsunami is observed
                
                Challenges and improvements after 2011:
                
                The 2011 Tohoku earthquake and tsunami revealed several weaknesses in Japan's warning system:
                
                1. Underestimation of maximum tsunami heights
                   - Warning initially predicted 3-6 meter waves
                   - Actual tsunami heights reached over 40 meters in some areas
                
                2. Power and communication failures
                   - Many warning systems lost power
                   - Communication networks were overwhelmed
                
                Improvements since 2011:
                
                1. Better modeling of tsunami generation
                   - Consideration of submarine landslides
                   - Improved bathymetric data
                
                2. Redundant communication systems
                   - Satellite-based backup systems
                   - Battery and generator backups
                
                3. Enhanced public education
                   - Emphasis on immediate evacuation regardless of warning levels
                   - Regular drills and exercises
                """)
            
            self.logger.info("Sample text files created successfully.")
    
    def build(self):
        """
        Build the chatbot by processing the knowledge base.
        """
        if self.is_built:
            self.logger.info("Chatbot is already built. Skipping...")
            return
            
        self.logger.info("Building the Text-Only Earthquake Chatbot...")
        self.logger.info("This may take some time...")
        
        try:
            self.pipeline.build()
            self.is_built = True
            self.logger.info("Chatbot successfully built!")
        except Exception as e:
            self.logger.error(f"Error building the chatbot: {str(e)}")
            raise
    
    def ask(self, question):
        """
        Ask a question to the chatbot.
        """
        if not self.is_built:
            self.logger.info("Chatbot is not built yet. Building now...")
            self.build()
            
        self.logger.info(f"Received question: {question}")
        
        try:
            response = self.pipeline.generate(question)
            return response
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return f"I encountered an error: {str(e)}"
    
    def interactive_session(self):
        """
        Start an interactive chat session.
        """
        if not self.is_built:
            print("Building the Text-Only Earthquake Chatbot... This may take a few minutes.")
            self.build()
            
        print("\n===== Earthquake Expert Chatbot =====")
        print("Ask me anything about earthquakes, especially Japan's earthquakes,")
        print("prediction methods, or tsunami warning systems.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Earthquake Expert Chatbot. Goodbye!")
                break
                
            response = self.ask(user_input)
            print(f"\nEarthquake Expert: {response}")

# Example usage
if __name__ == "__main__":
    # Create and start the chatbot
    chatbot = TextOnlyEarthquakeChatbot(
        knowledge_base_path="./text_knowledge_base",
        reflection=2,
        vector_store_path="./earthquake_vector_store"
    )
    
    # Start an interactive session
    chatbot.interactive_session()
