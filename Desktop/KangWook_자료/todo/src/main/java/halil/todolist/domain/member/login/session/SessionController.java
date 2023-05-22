package halil.todolist.domain.member.login.session;

import halil.todolist.domain.member.dto.LoginDto;
import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.login.cookie.CookieService;
import halil.todolist.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

@Controller
@RequiredArgsConstructor
public class SessionController {

    private final MemberService memberService;
    private final SessionService sessionService;
    private final CookieService cookieService;              // HttpSession 확인 Bean 등록

    @GetMapping("/")
    public String index() {
        return "session/login";
    }

    /**
     *
     * @param signUpDto : 계정 생성
     * @return
     */
    @GetMapping("/session/signup")
    public String signUpForm(@ModelAttribute SignUpDto signUpDto) {
        // model.addAttribute("signUpDto", new SignUpDto());
        return "session/signup";
    }

    @PostMapping("/session/signup")
    public String signup(@ModelAttribute("signUpDto") SignUpDto signUpDto, Model model) {
        model.addAttribute("signUpDto", memberService.signUp(signUpDto));
        return "redirect:/session/login";
    }

    /**
     * 로그인
     * @param loginDto
     * @return
     */
    @GetMapping("/session/login")
    public String loginForm(@ModelAttribute LoginDto loginDto) {
        // model.addAttribute("loginDto", new LoginDto());
        return "session/login";
    }

    @PostMapping("/session/login")
    public String login(@ModelAttribute LoginDto loginDto,
                        HttpServletResponse response,
                        HttpServletRequest request) {
        Member member = sessionService.login(loginDto.getEmail(), loginDto.getPassword(), response);

        // 세션이 있으면 반환, 없으면 신규 세션 생성
        HttpSession session = request.getSession();
        // 세션에 Member 정보(email, password) 정보 보관
        session.setAttribute("SessionId", member);

        return "redirect:/todos";
    }

    @GetMapping("/session/get/{id}")
    public ResponseEntity getSession(HttpServletRequest request) {
        return ResponseEntity.ok(sessionService.getSession(request));
    }

    /**
     * @ HttpSession 을 사용
     */
    //@PostMapping("/session/login")
    public String loginHttpSession(@ModelAttribute LoginDto loginDto,
                        HttpServletResponse response,
                        HttpServletRequest request) {

        Member member = cookieService.login(response, loginDto.getEmail(), loginDto.getPassword());

        // 세션이 있으면 반환, 없으면 신규 세션 생성
        HttpSession session = request.getSession();
        // 세션에 Member 정보(email, password) 정보 보관
        session.setAttribute("SessionId", member);

        return "redirect:/todos";
    }

    @GetMapping("/logout")
    public String logout(HttpServletRequest request, HttpSession session) {
        // sessionService.expire(request);
        // session.invalidate();
        sessionService.logout(request);
        return "redirect:/session/login";
    }
}
