# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container (like cd)

# Copy the script into the container at /usr/src/app
COPY ./scraper /scraper

# Install any needed packages specified in requirements.txt
RUN pip install beautifulsoup4 pandas tqdm requests
WORKDIR /scraper
# Run bla.py when the container launches
CMD ["python","scrape.py"]
