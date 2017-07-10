#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "math.h"
#include <sys/time.h>



#define SAVEVIDEO

#define DEMO 1

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
#include <stdio.h>
#include <math.h>


#ifdef SAVEVIDEO
static CvVideoWriter *mVideoWriter;
#endif

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;
//Kyle 7/6/2017
static float *xcoords;
static float *ycoords;
static const int coords_size = 10000;
static int curr_coords_index = 0;
//

static int demo_delay = 0;
static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *last_avg2;
static float *last_avg;
static float *avg;
static float *testsize;
static float *movement;
static char **objectnames;
static image prev;
static image curr;
double demo_time;

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;
    int first_index = 0;

    layer l = net.layers[net.n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = last_avg2;
    if(demo_delay == 0) l.output = avg;
    if(l.type == DETECTION){
      //Creating boxes according to the box
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
      //
        get_region_boxes(l, buff[0].w, buff[0].h, net.w, net.h, demo_thresh, probs, boxes, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    image display = buff[(buff_index+2) % 3];

    curr = display;


//TODO Don't pass a constant.
//Try to get the x coordinates and y coordinates of the detected boxes and store them into movement array.
    trackperson(demo_detections,.5,boxes,probs,demo_classes,movement,testsize,demo_names, objectnames);
//Draw the detections of different objects
    draw_detections(display, demo_detections, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net.w, net.h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 10){
        if(demo_delay == 0) demo_delay = 60;
        else if(demo_delay == 5) demo_delay = 0;
        else if(demo_delay == 60) demo_delay = 5;
        else demo_delay = 0;
    } else if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    demo_delay = delay;
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);

        #ifdef SAVEVIDEO
        if(cap){
            int mfps = cvGetCaptureProperty(cap,CV_CAP_PROP_FPS);
            mVideoWriter=cvCreateVideoWriter("Output.avi",CV_FOURCC('M','J','P','G'),mfps,cvSize(cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH),cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT)),1);
        }
#endif
    }else{
        cap = cvCaptureFromCAM(cam_index);

        #ifdef SAVEVIDEO
        if(cap){
            int mfps = cvGetCaptureProperty(cap,CV_CAP_PROP_FPS);
            mVideoWriter=cvCreateVideoWriter("Output.avi",CV_FOURCC('M','J','P','G'),mfps,cvSize(cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH),cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT)),1);
        }
#endif

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }


    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    int j;
//create a array to store the coordinates of
    movement = (float*)calloc(sizeof(float*)*1000,1);
    testsize = (float*)calloc(sizeof(float*)*1000,1);
    objectnames = (char**) calloc(sizeof(char*) * 1000, 1);
    for (int i = 0;i<4000;i++) movement[i] = 0.0;
    avg = (float *) calloc(l.outputs, sizeof(float));
    last_avg  = (float *) calloc(l.outputs, sizeof(float));
    last_avg2 = (float *) calloc(l.outputs, sizeof(float));


    //Kyle 7/6/2017
    xcoords = (int*)calloc(coords_size, sizeof(int));
    ycoords = (int*)calloc(coords_size, sizeof(int));
    //
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[1] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[2] = letterbox_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL);
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = get_wall_time();
//currx and curry store the current location of the tracked object
    float currx;
    float curry;
    float currw;
    float currh;
//prevx and prevy store the previous location of the tracked object
    float prevx;
    float prevy;
    float prevw;
    float prevh;
    float dangerousw;
    float dangeroush;
    char* prevname;
    char* currname;
    int draw = 0;
    float dx = 0;
    float dy = 0;
    int compare = 0;
    float min = 0;
    int countframe = 0;
    int dangerousi = 0;
    while(!demo_done){
        printf("count is %i \n", count);

        dangerousw = 0.0;
        dangeroush = 0.0;
        dangerousi = 0;
        draw = 1;

        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        if (count > 1){
        //Compare is a boolean that shows whether there is a previous frame. If not compare = 0, and we only need to update the current location
          if (compare == 0){
            for (int i = 0; i < (int)(movement[0]); i++){
              if (strcmp(objectnames[2*i+1], "person") == 0 ){
                currx = movement[2*i+1];
                curry = movement[2*i+2];
                currw = testsize[2*i+1];
                currh = testsize[2*i+2];
                currname = objectnames[2*i+1];
                break;
              }
            }
            prevx = currx;
            prevw = currw;
            prevh = currh;
            prevy = curry;
            prev = curr;
            prevname = currname;
//Kyle 7/6/2017
            xcoords[curr_coords_index] = prevx;
            ycoords[curr_coords_index] = prevy;
            curr_coords_index ++;
//
        //Then we update compare to 1
            compare = 1;
          }
          else if ((int)movement[0] != 0){
            // int index = 0;
            // for (int i = 0; i < (int)(movement[0]); i++){
            //   if (strcmp(objectnames[2*i+1], "person") == 0){
            //     currx = movement[2*i+1];
            //     curry = movement[2*i+2];
            //     currw = testsize[2*i+1];
            //     currh = testsize[2*i+2];
            //     currname = objectnames[2*i+1];
            //     index = i;
            //     break;
            //   }
            // }
            currx = movement[1];
            curry = movement[2];
            currw = testsize[1];
            currh = testsize[2];
            currname = objectnames[1];
            float minavg;
            float curravg = (currw+currh)*0.5;
            float prevavg = (prevw+prevh)*0.5;
            min = (prevx-currx)*(prevx-currx) + (prevy-curry)*(prevy-curry) + (curravg-prevavg)*(curravg - prevavg);
          //Find the object that has a minimum distance from the previous object
            for (int i = 0; i < (int)(movement[0]); i++){
              minavg = (testsize[2*i+1] + testsize[2*i+2])*0.5;
              if (min > (prevavg - minavg)*(prevavg - minavg) + (prevx-movement[2*i+1])*(prevx-movement[2*i+1]) + (prevy-movement[2*i+2])*(prevy-movement[2*i+2])){
                min = (prevx-movement[2*i+1])*(prevx-movement[2*i+1]) + (prevy-movement[2*i+2])*(prevy-movement[2*i+2]);
          //updating current x and y coordinates and the width and height of the box
                currx = movement[2*i+1];
                curry = movement[2*i+2];
                currw = testsize[2*i+1];
                currh = testsize[2*i+2];
                currname = objectnames[2*i+1];
              }
            }
            float the_min = fmin(currw, prevw);
            printf("the first one is %f \n", currw);
            printf("the second one is %f \n", prevw);
            printf("the width is %f \n", fmin(currw, prevw));
            printf("the image width is %i \n", buff[(buff_index + 1)%3].w);



            printf("the real diff is %f \n", compare_image(prev, curr, prevx, prevy, prevw, prevh, currx, curry, currw, currh));
            printf("the first diff is %f \n", compare_image(prev, curr, prevx, prevy, prevw, prevh, movement[1], movement[2], testsize[1], testsize[2]));
          // we did not find the object
          if (((fabs(currw - prevw) >  0.8* fmin(prevw, currw) || fabs(currh-  prevh) >  0.8*fmin(prevh, currh)) || fabs(currx-prevx) > currw || fabs(curry - prevy) > currh || strcmp(currname, prevname) != 0 || (int)(movement[0]) == 0) && countframe < 29){
            printf("loss!!!!!!!!!!!!!!!!!!!!!! \n");
            printf("dx is %f \n", dx);
            printf("dy is %f \n", dy);
            currx = prevx+dx;
            curry = prevy+dy;


            printf("xcoords[curr_coords_index-1] is %f \n", xcoords[curr_coords_index-1]);
              // currx = prediction(xcoords, curr_coords_index);
              // curry = prediction(ycoords, curr_coords_index);

            currw = prevw;
            currh = prevh;
            currname = prevname;
            countframe++;
            draw = 0;

          }
          printf("countframe is %i \n", countframe);
        }
        else {
          draw = 0;
        }


        xcoords[curr_coords_index] = prevx;
        ycoords[curr_coords_index] = prevy;
        dx = currx - prevx;
        dy = curry - prevy;
        prevx = currx;
        prevy = curry;
        prevw = currw;
        prevh = currh;
        prevname = currname;
        curr_coords_index ++;
        prev = curr;
        // printf("draw is %i \n", draw);
        // printf("diff is %f \n", currw-prevw);
        if (draw == 1){
          countframe = 0;
          if (fabs((currx - prevx)*buff[(buff_index + 1)%3].w) < .5 && fabs((curry- prevy)*buff[(buff_index + 1)%3].h ) <.5) {
            printf("relative movement is very small! \n");
            // TODO Do not draw box when no object is on the frame
            draw_box_width(buff[(buff_index + 1)%3], (int)((currx-currw/2)*buff[(buff_index + 1)%3].w), (int)((curry-currh/2)*buff[(buff_index + 1)%3].h),(int)((currx+currw/2)*buff[(buff_index + 1)%3].w), (int)((curry+currh/2)*buff[(buff_index + 1)%3].h), buff[(buff_index + 1)%3].w/80, 0.175, 0.0, 0.5);
          }
          else{
            draw_box_width(buff[(buff_index + 1)%3], (int)((currx-currw/2)*buff[(buff_index + 1)%3].w), (int)((curry-currh/2)*buff[(buff_index + 1)%3].h),(int)((currx+currw/2)*buff[(buff_index + 1)%3].w), (int)((curry+currh/2)*buff[(buff_index + 1)%3].h), buff[(buff_index + 1)%3].w/80, 0.0, 0.0, 0.5);
          // draw_box_width(buff[(buff_index + 1)%3], (int)(currx*buff[(buff_index + 1)%3].w)-testsize[currx]/2, (int)(curry*buff[(buff_index + 1)%3].h)-testsize[curry]/2,(int)(currx*buff[(buff_index + 1)%3].w)+testsize[currx]/2, (int)(curry*buff[(buff_index + 1)%3].h)+testsize[curry]/2, 3, 1.0, 1.0, 1.0);
          // draw_box_width(buff[(buff_index + 1)%3], (int)(currx*buff[(buff_index + 1)%3].w)-25, (int)(curry*buff[(buff_index + 1)%3].h)-25,(int)(currx*buff[(buff_index + 1)%3].w)+25, (int)(curry*buff[(buff_index + 1)%3].h)+25, 3, 0.0, 0.0, 0.0);
          // }
          }
        // printf("x_movement is %i \f", (currx - prevx)*buff[(buff_index + 1)%3].w);
        // printf("current x_position is %i\n" ,(int)(currx*buff[(buff_index + 1)%3].w));
        // printf("y_movement is %f \n", (curry- prevy)*buff[(buff_index + 1)%3].h);
        // printf("current y_position is %i \n", (int)(curry*buff[(buff_index + 1)%3].h));
        //update the previous x and y coordinates
        // printf("the width of the image is %i\n", buff[(buff_index + 1)%3].w);
        // printf("the height of the image is %i\n", buff[(buff_index + 1)%3].h);
        //
        // printf("number of object is %f \n", movement[0]);

        //Kyle 7/6/2017
          draw_tracers(buff[(buff_index + 1)%3], curr_coords_index, xcoords, ycoords);

        }
        //
      }
        if(!prefix){

          #ifdef SAVEVIDEO
                save_video(buff[(buff_index + 1)%3],mVideoWriter);
#endif

            if(count % (demo_delay+1) == 0){
                fps = 1./(get_wall_time() - demo_time);
                demo_time = get_wall_time();
                float *swap = last_avg;
                last_avg  = last_avg2;
                last_avg2 = swap;
                memcpy(last_avg, avg, l.outputs*sizeof(float));
            }
            display_in_thread(0);
        }else{
            char name[256];
            // sprintf(name, "%s_%08d", prefix, count);

            #ifdef SAVEVIDEO
                            save_video(buff[(buff_index + 1)%3],mVideoWriter);
            #else

            save_image(buff[(buff_index + 1)%3], name);

            #endif
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
    printf("the number of centroid is %i \n", curr_coords_index);
    free(movement);
    free(testsize);
    free(objectnames);
    //Kyle 7/6/2017
    free(xcoords);
    free(ycoords);
    // //
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
