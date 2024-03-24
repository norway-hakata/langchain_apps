# Langchain Apps

Langchain Apps is an application that uses Langchain to summarize YouTube videos and website content. To get started, follow the instructions below to set up and run the application locally.

## Initial Setup

1. Clone this repository to your local machine:
```shell-session
git clone https://github.com/norway-hakata/langchain_apps.git
```

2. Navigate to the cloned directory:
```shell-session
cd langchain_apps
```


## Environment Variable Setup

Before running the application, you need to set your OpenAI API key. Follow the steps below to create and set up your `.env` file.

1. Create a `.env` file at the root of the directory.
2. Add the following content to your `.env` file (replace `your_openai_api_key_here` with your actual API key):
```.env
OPENAI_API_KEY="your_openai_api_key_here"
```
3. Ensure that your `.env` file is not published. It is recommended to add this file to your `.gitignore`.

## Launching the Application with Docker Compose

Make sure Docker and Docker Compose are installed on your system. If not, please set them up by following the [official Docker documentation](https://docs.docker.com/get-docker/).

Then, build and start the application by running:
```shell-session
docker-compose up
```

This command builds the necessary Docker image and starts the application. The `--build` flag ensures that the image is built based on the latest source code.

## Accessing the Application

After the application has started, open your browser and go to `localhost:8501`. This will display the Langchain Apps interface.

## Usage

The application provides fields for entering YouTube video or website URLs. Enter the appropriate URL and follow the prompts to retrieve a summary of the content.

Thank you for using Langchain Apps!
