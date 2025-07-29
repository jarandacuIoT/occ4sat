from abc import ABC, abstractmethod


class Camera(ABC):

    @abstractmethod
    def configure_video(self):
        pass

    @abstractmethod
    def record_video(self, folder_name, file_name, duration):
        pass
    

class ConfigCamera(ABC):
    
    @abstractmethod
    def configure(self):
        pass    

