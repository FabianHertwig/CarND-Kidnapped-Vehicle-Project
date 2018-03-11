/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {


    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);


    for (int i = 0; i < num_particles; ++i) {

        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        particles.push_back(particle);
        weights.push_back(1.0);

    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    for (auto &particle : particles) {

        if (fabs(yaw_rate) > MINIMAL_YAW_RATE) {
            particle.x +=
                    velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y +=
                    velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            particle.theta += yaw_rate * delta_t;
        } else {
            particle.x = particle.x + velocity * delta_t * cos(particle.theta);
            particle.y = particle.y + velocity * delta_t * sin(particle.theta);
        }

        normal_distribution<double> dist_x(particle.x, std_pos[0]);
        normal_distribution<double> dist_y(particle.y, std_pos[1]);
        normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);

    }

}

std::vector<LandmarkObs>
ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    std::vector<LandmarkObs> result;
    LandmarkObs closest_prediction = LandmarkObs();

    for (LandmarkObs observed : observations) {

        double shortest_distance = 10E10;
        for (LandmarkObs prediction : predicted) {

            double distance = dist(observed.x, observed.y, prediction.x, prediction.y);
            if (distance <= shortest_distance) {
                shortest_distance = distance;
                closest_prediction = prediction;
            }
        }
        result.push_back(closest_prediction);
    }

    return result;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    for (int i = 0; i < particles.size(); i++) {

        Particle particle = particles[i];

        std::vector<LandmarkObs> transformed_observations;
        for (auto observation : observations) {

            LandmarkObs transformed_observation{};
            transformed_observation.x =
                    particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
            transformed_observation.y =
                    particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);
            transformed_observation.id = observation.id;

            transformed_observations.push_back(transformed_observation);
        }


        std::vector<LandmarkObs> observations_in_range;
        for (auto map_landmark: map_landmarks.landmark_list) {

            double distance = dist(particle.x, particle.y, map_landmark.x_f, map_landmark.y_f);
            if (distance <= sensor_range) {
                LandmarkObs landmark_in_range{};
                landmark_in_range.id = map_landmark.id_i;
                landmark_in_range.x = map_landmark.x_f;
                landmark_in_range.y = map_landmark.y_f;
                observations_in_range.push_back(landmark_in_range);
            }
        }

        std::vector<LandmarkObs> associated_landmarks = dataAssociation(observations_in_range,
                                                                        transformed_observations);

        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double weight = 1.0;
        for (unsigned long k = 0; k < associated_landmarks.size(); k++) {

            double dx = transformed_observations.at(k).x - associated_landmarks.at(k).x;
            double dy = transformed_observations.at(k).y - associated_landmarks.at(k).y;

            double gauss_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));
            double exponent = (dx * dx) / (2.0 * sig_x * sig_x) + (dy * dy) / (2.0 * sig_y * sig_y);

            weight *= gauss_norm * exp(-exponent);

        }
        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
    std::discrete_distribution<int> discreteDistribution(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles(num_particles);

    for (int i = 0; i < num_particles; i++) {
        int sampled_index = discreteDistribution(gen);
        resampled_particles[i] = particles[sampled_index];
    }

    particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
