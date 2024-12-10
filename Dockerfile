FROM nginx:alpine

#set working directory
WORKDIR /usr/share/nginx/html

#copy files
COPY . .

#expose port
EXPOSE 5000

#run command
CMD ["nginx", "-g", "daemon off;"]

