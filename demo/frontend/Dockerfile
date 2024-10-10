# Use alias build to be easier to refer this container elsewhere
FROM node:22-alpine as build

ENV REACT_APP_API=http://localhost

WORKDIR /app

# Copy package.json and package-lock.json to the container
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the entire project to the container
COPY . .

# Build the React.js app
RUN npm run build

# Set up production environment
FROM nginx:alpine

# Copy the built React.js app to the Nginx public directory
COPY --from=build /app/build /usr/share/nginx/html

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx/nginx.conf /etc/nginx/conf.d

# Expose the Nginx HTTP port
EXPOSE 80

# Start Nginx when the container starts
CMD ["nginx", "-g", "daemon off;"]
