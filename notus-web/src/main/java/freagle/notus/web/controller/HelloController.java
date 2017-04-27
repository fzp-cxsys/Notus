package freagle.notus.web.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * Created by a1477 on 2017/4/21.
 */
@Controller
public class HelloController {
    @RequestMapping("/index")
    public String index(){
        return "index.html";
    }
}
