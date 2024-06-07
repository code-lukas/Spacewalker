FROM python:3.11-bookworm

COPY . .

RUN groupadd --gid 1000 pn && useradd --uid 1000 --gid pn --shell /bin/bash --create-home pn
RUN apt update && apt install -y ca-certificates curl gnupg
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
ENV NODE_MAJOR=20
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
RUN apt update && apt install nodejs -y

RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" > /etc/apt/sources.list.d/yarn.list
RUN wget -qO- https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN apt update
RUN apt upgrade -y
RUN apt install -y yarn build-essential
RUN pip install -U pip
RUN rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
# pip packages
RUN pip install -r ./requirements.txt

# npm installations
RUN npm install -g npm@latest
RUN npm i

EXPOSE 8080

#CMD ["python3", "-u", "manage.py", "migrate"]
#CMD ["python3", "-u", "manage.py", "runserver", "0.0.0.0:8080"]
