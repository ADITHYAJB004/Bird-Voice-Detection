# Project: Bird Vocalization Detection
The Bird Vocalization Detection is a project focused on AI based Bird Family Detection based on the bird audio.

## Project Documentation

### Requirements
- Python : 3.12.5+

### Process Followed
- Environment Setup
    > python -m venv .env

    > .env\Scripts\activate

- Install Django
    > Reference: https://www.djangoproject.com/download/

    > py -m pip install Django==5.1.5

- Create Requirements Doc
    > pip freeze> requirements.txt

- Create New Django Project
    > django-admin startproject bird_vocal_detection 

- Create New Django App
    > We need two apps
    > - **bird_detection_interface**: User Interface for User to Post Images.
    > - **bird_chat_bot**: Chat interface for user to ask questions about different birds.

    Commands
    > python manage.py startapp bird_detection_interface

    > python manage.py startapp bird_chat_bot

- Project Outline Upto Now
    ![project_outline](ReadMe_Static\Images\initial_project_outline.png)

- Logo From: 
    > Link: https://www.freepik.com/free-vector/bird-colorful-logo-gradient-vector_28267842.htm#fromView=search&page=1&position=42&uuid=48486fdb-6a03-4a6b-a420-76c46ae23c3c&new_detail=true&query=Bird+logo